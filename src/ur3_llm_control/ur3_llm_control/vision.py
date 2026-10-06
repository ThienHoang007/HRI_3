"""Camera ảo nhìn từ trên xuống + nhận dạng khối bằng OpenCV (không phụ thuộc ROS).

Camera ảo là camera RGB-D (ảnh màu + ảnh độ sâu) gắn cố định phía trên bàn, nhìn thẳng xuống.
Bộ mô phỏng vẽ ảnh từ trạng thái thật của thế giới; phần nhận dạng chỉ dùng ảnh + nội/ngoại tham số camera.
"""
import math

import cv2
import numpy as np

# Màu BGR dùng để vẽ (đủ bão hoà để tách được bằng ngưỡng HSV trong scene.yaml)
DRAW_BGR = {
    'red': (40, 30, 220),
    'yellow': (30, 215, 235),
    'green': (60, 180, 40),
    'blue': (210, 90, 20),
    'purple': (170, 40, 140),
}
TABLE_BGR = (150, 170, 185)
FLOOR_BGR = (60, 60, 60)
ZONE_BGR = (245, 245, 245)


# Camera nhìn thẳng xuống: trục x_cam = -y_world, y_cam = -x_world, z_cam = -z_world
DOWN_ROTATION = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
DOWN_QUATERNION = (0.7071067811865476, -0.7071067811865476, 0.0, 0.0)   # x, y, z, w


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class CameraModel:
    """Camera pinhole. R: ma trận xoay camera_optical -> world, t: vị trí camera trong world."""

    def __init__(self, width, height, fx, fy, cx=None, cy=None, R=DOWN_ROTATION, t=(0, 0, 1)):
        self.w, self.h = int(width), int(height)
        self.fx, self.fy = float(fx), float(fy)
        self.cx = self.w / 2.0 if cx is None else float(cx)
        self.cy = self.h / 2.0 if cy is None else float(cy)
        self.R = np.asarray(R, dtype=float)
        self.pos = np.asarray(t, dtype=float)

    @classmethod
    def from_config(cls, cam_cfg):
        return cls(cam_cfg['width'], cam_cfg['height'], cam_cfg['fx'], cam_cfg['fy'],
                   t=cam_cfg['position'])

    def K(self):
        return [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]

    def to_cam(self, x, y, z):
        return self.R.T @ (np.array([x, y, z], dtype=float) - self.pos)

    def project(self, x, y, z):
        """world -> pixel (u, v)."""
        pc = self.to_cam(x, y, z)
        return self.cx + self.fx * pc[0] / pc[2], self.cy + self.fy * pc[1] / pc[2]

    def depth_of(self, x, y, z):
        return float(self.to_cam(x, y, z)[2])

    def unproject(self, u, v, d):
        """pixel + độ sâu d (theo trục quang) -> điểm world (x, y, z)."""
        pc = np.array([(u - self.cx) * d / self.fx, (v - self.cy) * d / self.fy, d])
        return self.R @ pc + self.pos


def _square(cam, x, y, z, half, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half)):
        pts.append(cam.project(x + c * dx - s * dy, y + s * dx + c * dy, z))
    return np.array(pts, dtype=np.float32)


def render(cam, cfg, blocks, rng=None):
    """Vẽ ảnh màu BGR + ảnh độ sâu (mét, float32) từ trạng thái thật của thế giới (RGB-D).

    blocks: {name: {'color', 'x', 'y', 'z', 'yaw'}} (z = tâm khối).
    """
    img = np.full((cam.h, cam.w, 3), FLOOR_BGR, dtype=np.uint8)
    depth = np.full((cam.h, cam.w), cam.depth_of(0, 0, -0.75), dtype=np.float32)  # sàn nhà
    t = cfg['table']
    top = t['center'][2] + t['size'][2] / 2.0
    tx, ty = t['center'][0], t['center'][1]
    corners = [cam.project(tx + sx * t['size'][0] / 2, ty + sy * t['size'][1] / 2, top)
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    cv2.fillPoly(img, [np.int32(corners)], TABLE_BGR)
    cv2.fillPoly(depth, [np.int32(corners)], cam.depth_of(tx, ty, top))

    for name, z in cfg['zones'].items():
        poly = _square(cam, z['center'][0], z['center'][1], top, z['size'] / 2.0)
        cv2.polylines(img, [np.int32(poly)], True, ZONE_BGR, 2, cv2.LINE_AA)
        u, v = poly[2]
        cv2.putText(img, name[-1].upper(), (int(u) + 4, int(v) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_BGR, 2, cv2.LINE_AA)

    # Đế robot (vòng tròn tối) để ảnh dễ đọc
    u0, v0 = cam.project(0.0, 0.0, 0.15)
    cv2.circle(img, (int(u0), int(v0)), int(cam.fx * 0.064 / cam.depth_of(0, 0, 0.15)), (45, 45, 45), -1)

    half = cfg['block_size'] / 2.0
    # Khối cao hơn vẽ sau (che khối thấp hơn)
    for b in sorted(blocks.values(), key=lambda b: b['z']):
        poly = _square(cam, b['x'], b['y'], b['z'] + half, half, b.get('yaw', 0.0))
        cv2.fillPoly(img, [np.int32(np.round(poly))], DRAW_BGR[b['color']], cv2.LINE_AA)
        cv2.fillPoly(depth, [np.int32(np.round(poly))], cam.depth_of(b['x'], b['y'], b['z'] + half))

    sigma = float(cfg['camera'].get('noise_sigma', 0.0))
    if sigma > 0:
        rng = rng or np.random.default_rng()
        noise = rng.normal(0.0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        depth = depth + rng.normal(0.0, 0.001, depth.shape).astype(np.float32)
    return img, depth


def detect(cam, cfg, img_bgr, depth):
    """Tách khối theo màu HSV trên ảnh màu, lấy độ cao từ ảnh depth.

    Trả về list {'color', 'x', 'y', 'z', 'u', 'v', 'area'} (z = tâm khối).
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    pcfg = cfg['perception']
    size = cfg['block_size']
    kernel = np.ones((3, 3), np.uint8)
    found = []
    for color, ranges in pcfg['colors'].items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < pcfg['min_area']:
                continue
            m = cv2.moments(c)
            u, v = m['m10'] / m['m00'], m['m01'] / m['m00']
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 255, -1)
            blob = cv2.erode(blob, kernel, iterations=2)   # bỏ viền (pixel lẫn nền)
            d = float(np.median(depth[blob > 0]))          # độ sâu mặt trên khối
            x, y, z_top = cam.unproject(u, v, d)
            # Góc xoay quanh trục z: lấy 1 cạnh của hình chữ nhật bao nhỏ nhất, đổi sang world
            box = cv2.boxPoints(cv2.minAreaRect(c))
            p0 = cam.unproject(box[0][0], box[0][1], d)
            p1 = cam.unproject(box[1][0], box[1][1], d)
            yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4   # khối vuông: đưa về [-45°, 45°)
            found.append({'color': color, 'x': float(x), 'y': float(y), 'z': float(z_top) - size / 2.0,
                          'yaw': yaw, 'u': u, 'v': v, 'area': area})
    return found


def zone_of(cfg, x, y):
    """Tên zone chứa điểm (x, y) hoặc None."""
    for name, z in cfg['zones'].items():
        h = z['size'] / 2.0
        if abs(x - z['center'][0]) <= h and abs(y - z['center'][1]) <= h:
            return name
    return None


def build_scene(cfg, detections, resting_z):
    """Gom kết quả detect thành trạng thái cảnh: vị trí từng khối + zone nào trống/bận."""
    objects = {}
    for d in sorted(detections, key=lambda d: -d['area']):
        name = f"{d['color']}_cube"
        if name in objects:      # cùng màu nhiều blob: giữ blob lớn nhất
            continue
        on_table = d['z'] < resting_z + cfg['block_size'] * 0.75
        objects[name] = {
            'x': round(d['x'], 4), 'y': round(d['y'], 4), 'z': round(d['z'], 4),
            'yaw': round(d['yaw'], 3),
            'on_table': bool(on_table),
            'zone': zone_of(cfg, d['x'], d['y']) if on_table else None,
        }
    zones = {z: None for z in cfg['zones']}
    for name, o in objects.items():
        if o['zone'] and zones[o['zone']] is None:
            zones[o['zone']] = name
    return {'objects': objects, 'zones': zones}


def find_free_position(cfg, objects, ignore=()):
    """Tìm điểm trống trên bàn: xa các khối khác, không nằm trong/ sát zone, robot với tới.

    objects: {name: {'x', 'y', ...}}. Trả về (x, y) hoặc None.
    """
    fs = cfg['free_space']
    others = [(o['x'], o['y']) for n, o in objects.items() if n not in ignore]
    zones = [tuple(z['center']) for z in cfg['zones'].values()]
    best, best_d = None, float('inf')
    px, py = fs['prefer']
    for x in np.arange(fs['x'][0], fs['x'][1] + 1e-9, fs['step']):
        for y in np.arange(fs['y'][0], fs['y'][1] + 1e-9, fs['step']):
            r = math.hypot(x, y)
            if not fs['min_radius'] <= r <= fs['max_radius']:
                continue
            if any(math.hypot(x - ox, y - oy) < fs['clearance'] for ox, oy in others):
                continue
            if any(math.hypot(x - zx, y - zy) < fs['zone_clearance'] for zx, zy in zones):
                continue
            d = math.hypot(x - px, y - py)
            if d < best_d:
                best, best_d = (round(float(x), 3), round(float(y), 3)), d
    return best
