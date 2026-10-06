"""world_sim: mô phỏng thế giới vật lý (thay cho Gazebo) + camera RGB-D ảo.

- Giữ vị trí THẬT của 5 khối trên bàn (đặt ban đầu theo scene.yaml).
- Vật lý kẹp tối giản: khi 2 ngón khép lại quanh một khối (khối nằm giữa 2 ngón, độ mở ngón
  <= nửa cạnh khối) thì khối bị kẹp và đi theo gripper_tcp (đọc từ TF). Khi ngón mở ra thì khối
  rơi thẳng xuống mặt đỡ bên dưới (bàn hoặc khối khác). Không có lệnh nào "dịch chuyển" khối
  trực tiếp: khối chỉ di chuyển khi robot thực sự kẹp và mang nó đi.
- Camera ảo gắn cố định phía trên bàn: phát /camera/color/image_raw, /camera/depth/image_raw,
  /camera/camera_info và TF world -> camera_optical_frame.
- Hiển thị bàn, zone, khối thật trong RViz qua MarkerArray /world_markers.
"""
import math

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from . import common as C
from .vision import DOWN_QUATERNION, CameraModel, quat_to_matrix, render

RGBA = {
    'red': (0.86, 0.12, 0.16), 'yellow': (0.92, 0.84, 0.12), 'green': (0.16, 0.70, 0.24),
    'blue': (0.08, 0.35, 0.82), 'purple': (0.55, 0.16, 0.67),
}


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


class WorldSim(Node):
    def __init__(self):
        super().__init__('world_sim')
        self.declare_parameter('scene_file', 'scene.yaml')
        self.declare_parameter('initial_yaw_noise', 0.35)
        self.declare_parameter('seed', 7)
        self.cfg = C.load_yaml(self.get_parameter('scene_file').value)
        self.size = self.cfg['block_size']
        self.rest_z = C.resting_z(self.cfg)
        self.table_z = C.table_surface_z(self.cfg)
        self.rng = np.random.default_rng(self.get_parameter('seed').value)
        self.cam = CameraModel.from_config(self.cfg['camera'])

        self.blocks = {}
        self.held = None          # tên khối đang bị kẹp
        self.held_rel = None      # (R, p) của khối trong frame gripper_tcp
        self.finger = None        # độ mở mỗi ngón (m)
        self.reset_world()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)  # TF không bị render camera chặn
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_camera_tf()

        self.create_subscription(JointState, '/joint_states', self._on_joints, 1)
        self.pub_color = self.create_publisher(Image, '/camera/color/image_raw', 2)
        self.pub_depth = self.create_publisher(Image, '/camera/depth/image_raw', 2)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', 2)
        self.pub_markers = self.create_publisher(MarkerArray, '/world_markers', 2)
        self.create_service(Trigger, '~/reset', self._on_reset)

        self.create_timer(0.02, self._step_physics)
        self.create_timer(1.0 / float(self.cfg['camera']['rate']), self._publish_camera,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(0.1, self._publish_markers)
        self.get_logger().info('world_sim sẵn sàng: ' + ', '.join(
            f"{n}@({b['x']:.2f},{b['y']:.2f})" for n, b in self.blocks.items()))

    # ------------------------------------------------------------------ world state
    def reset_world(self):
        noise = float(self.get_parameter('initial_yaw_noise').value)
        self.blocks = {}
        for name, o in self.cfg['objects'].items():
            init = o['initial']
            xy = self.cfg['zones'][init]['center'] if isinstance(init, str) else init
            self.blocks[name] = {'color': o['color'], 'x': float(xy[0]), 'y': float(xy[1]),
                                 'z': self.rest_z, 'yaw': float(self.rng.uniform(-noise, noise))}
        self.held = None
        self.held_rel = None

    def _on_reset(self, _req, resp):
        self.reset_world()
        resp.success = True
        resp.message = 'đã đặt lại vị trí các khối'
        return resp

    def _on_joints(self, msg):
        if 'finger_joint' in msg.name:
            self.finger = msg.position[msg.name.index('finger_joint')]

    def _tcp_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('world', 'gripper_tcp', Time())
        except TransformException:
            return None
        q = t.transform.rotation
        p = t.transform.translation
        return quat_to_matrix(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z])

    def _support_z(self, name, x, y):
        """Độ cao tâm khối khi rơi xuống tại (x, y): trên bàn hoặc trên khối khác."""
        z = self.rest_z
        for other, b in self.blocks.items():
            if other == name:
                continue
            if abs(b['x'] - x) < self.size * 0.9 and abs(b['y'] - y) < self.size * 0.9:
                z = max(z, b['z'] + self.size)
        return z

    def _step_physics(self):
        tcp = self._tcp_pose()
        if tcp is None or self.finger is None:
            return
        R, p = tcp
        half = self.size / 2.0
        if self.held is None:
            if self.finger > half + 0.003:
                return
            # Khối nào đang nằm giữa 2 ngón khi ngón khép lại?
            for name, b in self.blocks.items():
                local = R.T @ (np.array([b['x'], b['y'], b['z']]) - p)
                rel_yaw = (b['yaw'] - yaw_of(R) + math.pi / 4) % (math.pi / 2) - math.pi / 4
                if (abs(local[0]) < 0.012 and abs(local[1]) < 0.008 and abs(local[2]) < 0.015
                        and abs(rel_yaw) < 0.2):
                    self.held = name
                    Rb = quat_to_matrix(0, 0, math.sin(b['yaw'] / 2), math.cos(b['yaw'] / 2))
                    self.held_rel = (R.T @ Rb, local)
                    self.get_logger().info(f'gripper đã kẹp {name}')
                    break
        else:
            b = self.blocks[self.held]
            if self.finger > half + 0.002:          # ngón mở -> khối rơi xuống
                b['z'] = self._support_z(self.held, b['x'], b['y'])
                self.get_logger().info(f"đã thả {self.held} tại ({b['x']:.3f}, {b['y']:.3f}, {b['z']:.3f})")
                self.held = None
                self.held_rel = None
                return
            Rrel, prel = self.held_rel
            pos = R @ prel + p
            b['x'], b['y'], b['z'] = float(pos[0]), float(pos[1]), float(pos[2])
            b['yaw'] = yaw_of(R @ Rrel)

    # ------------------------------------------------------------------ camera
    def _publish_camera_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = self.cfg['camera']['frame_id']
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = \
            [float(v) for v in self.cfg['camera']['position']]
        q = DOWN_QUATERNION
        t.transform.rotation.x, t.transform.rotation.y = q[0], q[1]
        t.transform.rotation.z, t.transform.rotation.w = q[2], q[3]
        self.static_tf.sendTransform(t)

    def _publish_camera(self):
        stamp = self.get_clock().now().to_msg()
        frame = self.cfg['camera']['frame_id']
        color, depth = render(self.cam, self.cfg, {k: dict(v) for k, v in self.blocks.items()}, self.rng)

        img = Image()
        img.header.stamp, img.header.frame_id = stamp, frame
        img.height, img.width = color.shape[:2]
        img.encoding, img.step = 'bgr8', color.shape[1] * 3
        img.data = color.tobytes()
        self.pub_color.publish(img)

        dimg = Image()
        dimg.header.stamp, dimg.header.frame_id = stamp, frame
        dimg.height, dimg.width = depth.shape
        dimg.encoding, dimg.step = '32FC1', depth.shape[1] * 4
        dimg.data = depth.astype(np.float32).tobytes()
        self.pub_depth.publish(dimg)

        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, frame
        info.width, info.height = self.cam.w, self.cam.h
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = self.cam.K()
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [self.cam.fx, 0.0, self.cam.cx, 0.0, 0.0, self.cam.fy, self.cam.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)

    # ------------------------------------------------------------------ RViz
    def _marker(self, mid, ns, mtype, xyz, scale, rgba, yaw=0.0):
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(v) for v in xyz]
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.scale.x, m.scale.y, m.scale.z = [float(v) for v in scale]
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in rgba]
        m.lifetime = Duration(sec=1)
        return m

    def _publish_markers(self):
        arr = MarkerArray()
        t = self.cfg['table']
        arr.markers.append(self._marker(0, 'table', Marker.CUBE, t['center'], t['size'], (0.72, 0.62, 0.50, 1.0)))
        for i, (name, z) in enumerate(sorted(self.cfg['zones'].items())):
            c = z['center']
            arr.markers.append(self._marker(i, 'zones', Marker.CUBE, (c[0], c[1], self.table_z + 0.0005),
                                            (z['size'], z['size'], 0.001), (0.95, 0.95, 0.95, 1.0)))
            label = self._marker(i, 'zone_labels', Marker.TEXT_VIEW_FACING,
                                 (c[0] + z['size'] / 2 + 0.03, c[1], self.table_z + 0.01),
                                 (0, 0, 0.035), (1, 1, 1, 1))
            label.text = name.upper().replace('_', ' ')
            arr.markers.append(label)
        for i, (name, b) in enumerate(sorted(self.blocks.items())):
            arr.markers.append(self._marker(i, 'blocks', Marker.CUBE, (b['x'], b['y'], b['z']),
                                            (self.size,) * 3, RGBA[b['color']] + (1.0,), b['yaw']))
        self.pub_markers.publish(arr)


def main():
    rclpy.init()
    node = WorldSim()
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
