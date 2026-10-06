"""perception_node: nhận dạng khối và trạng thái zone từ camera RGB-D (OpenCV).

Sub:  /camera/color/image_raw (bgr8), /camera/depth/image_raw (32FC1), /camera/camera_info
TF:   world -> frame của camera (lấy ngoại tham số)
Pub:  /perception/scene (std_msgs/String, JSON) và /perception/debug_image (ảnh có chú thích)
Srv:  /perception/detect (std_srvs/Trigger) - chờ 1 ảnh MỚI rồi trả JSON:
      {"objects": {name: {x, y, z, yaw, on_table, zone}}, "zones": {zone_a: name|null, ...}}
"""
import json
import threading

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from . import common as C
from .vision import CameraModel, build_scene, detect, quat_to_matrix


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception')
        self.declare_parameter('scene_file', 'scene.yaml')
        self.cfg = C.load_yaml(self.get_parameter('scene_file').value)
        self.rest_z = C.resting_z(self.cfg)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.info = None
        # color và depth đến lệch nhau (ảnh depth lớn hơn) -> ghép cặp theo đúng stamp chụp
        self.depths, self.colors = {}, {}
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.latest = None          # (thời điểm CHỤP ảnh [ns], scene dict)

        sensors = MutuallyExclusiveCallbackGroup()
        self.create_subscription(CameraInfo, '/camera/camera_info', self._on_info, 2, callback_group=sensors)
        self.create_subscription(Image, '/camera/depth/image_raw', self._on_depth, 5, callback_group=sensors)
        self.create_subscription(Image, '/camera/color/image_raw', self._on_color, 5, callback_group=sensors)
        self.pub_scene = self.create_publisher(String, '/perception/scene', 2)
        self.pub_debug = self.create_publisher(Image, '/perception/debug_image', 2)
        self.create_service(Trigger, '/perception/detect', self._on_detect,
                            callback_group=ReentrantCallbackGroup())
        self.get_logger().info('perception sẵn sàng (service /perception/detect)')

    def _on_info(self, msg):
        self.info = msg

    def _on_depth(self, msg):
        stamp = Time.from_msg(msg.header.stamp).nanoseconds
        self.depths[stamp] = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        self._process_pending()

    def _camera(self, frame_id):
        t = self.tf_buffer.lookup_transform('world', frame_id, Time())
        q, p = t.transform.rotation, t.transform.translation
        k = self.info.k
        return CameraModel(self.info.width, self.info.height, k[0], k[4], k[2], k[5],
                           R=quat_to_matrix(q.x, q.y, q.z, q.w), t=(p.x, p.y, p.z))

    def _on_color(self, msg):
        self.colors[Time.from_msg(msg.header.stamp).nanoseconds] = msg
        self._process_pending()

    def _process_pending(self):
        common = sorted(set(self.colors) & set(self.depths))
        if not common:
            self._trim()
            return
        stamp = common[-1]
        msg, depth = self.colors.pop(stamp), self.depths.pop(stamp)
        # bỏ các khung cũ hơn khung vừa ghép
        for d in (self.colors, self.depths):
            for k in [k for k in d if k <= stamp]:
                del d[k]
        if self.info is None:
            return
        try:
            cam = self._camera(msg.header.frame_id)
        except TransformException as e:
            self.get_logger().warn(f'chưa có TF camera: {e}', throttle_duration_sec=5.0)
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        dets = detect(cam, self.cfg, img, depth)
        scene = build_scene(self.cfg, dets, self.rest_z)
        with self.cond:
            self.latest = (stamp, scene)
            self.cond.notify_all()
        self.pub_scene.publish(String(data=json.dumps(scene)))
        self._publish_debug(msg, img, dets, scene)

    def _trim(self, keep=5):
        for d in (self.colors, self.depths):
            for k in sorted(d)[:-keep]:
                del d[k]

    def _publish_debug(self, msg, img, dets, scene):
        out = img.copy()
        for d in dets:
            u, v = int(d['u']), int(d['v'])
            cv2.circle(out, (u, v), 4, (255, 255, 255), -1)
            cv2.putText(out, f"{d['color']} ({d['x']:.2f},{d['y']:.2f},{d['z']:.2f})", (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y = 22
        for z, o in sorted(scene['zones'].items()):
            cv2.putText(out, f"{z}: {o or 'EMPTY'}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            y += 24
        dbg = Image()
        dbg.header = msg.header
        dbg.height, dbg.width = out.shape[:2]
        dbg.encoding, dbg.step = 'bgr8', out.shape[1] * 3
        dbg.data = out.tobytes()
        self.pub_debug.publish(dbg)

    def _on_detect(self, _req, resp):
        # Chỉ chấp nhận ảnh được CHỤP sau thời điểm gọi service (không dùng ảnh cũ trong hàng đợi)
        t0 = self.get_clock().now().nanoseconds
        with self.cond:
            ok = self.cond.wait_for(lambda: self.latest is not None and self.latest[0] > t0, timeout=5.0)
            scene = self.latest[1] if ok else None
        if scene is None:
            resp.success = False
            resp.message = 'không nhận được ảnh camera'
        else:
            resp.success = True
            resp.message = json.dumps(scene)
        return resp


def main():
    rclpy.init()
    node = PerceptionNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
