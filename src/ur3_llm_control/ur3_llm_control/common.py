"""Hằng số dùng chung, chuẩn hoá tên, đọc cấu hình và ánh xạ theo MSSV."""
import os
import re
import unicodedata

import yaml

SKILLS = {
    # skill: (tham số bắt buộc, tham số tùy chọn)
    'home': ((), ()),
    'open_gripper': ((), ()),
    'close_gripper': ((), ()),
    'detect_objects': ((), ()),
    'check_zone': (('zone',), ()),
    'find_object': (('object',), ()),
    'find_free_position': ((), ('object',)),
    'move_above': ((), ('object', 'zone')),
    'move_to_zone': (('zone',), ()),
    'pick': (('object',), ()),
    'place': (('object', 'zone'), ()),
}

OBJECTS = ['red_cube', 'yellow_cube', 'blue_cube', 'green_cube', 'purple_cube']
ZONES = ['zone_a', 'zone_b', 'zone_c']
TEMP = 'temporary_position'

COLOR_OF = {o: o.split('_')[0] for o in OBJECTS}

# Mã trạng thái trả về của skill / validator
SUCCESS = 'SUCCESS'
FAILED = 'FAILED'
INVALID_OBJECT = 'INVALID_OBJECT'
INVALID_ZONE = 'INVALID_ZONE'
INVALID_SKILL = 'INVALID_SKILL'
PLANNING_FAILED = 'PLANNING_FAILED'
ZONE_OCCUPIED = 'ZONE_OCCUPIED'

# Bảng ánh xạ theo đề: P -> màu của zone A, B, C
STUDENT_TABLE = {
    0: ('red', 'yellow', 'blue'),
    1: ('red', 'blue', 'yellow'),
    2: ('yellow', 'red', 'blue'),
    3: ('yellow', 'blue', 'red'),
    4: ('blue', 'red', 'yellow'),
    5: ('blue', 'yellow', 'red'),
}

_ALIASES_OBJ = {
    'red_cube': ['red', 'do', 'mau do'],
    'yellow_cube': ['yellow', 'vang', 'mau vang'],
    'blue_cube': ['blue', 'xanh duong', 'xanh lam', 'xanh da troi', 'mau xanh duong'],
    'green_cube': ['green', 'xanh la', 'xanh luc', 'mau xanh la'],
    'purple_cube': ['purple', 'violet', 'tim', 'mau tim'],
}


def normalize(text):
    """Chữ thường, bỏ dấu tiếng Việt (đ -> d), gom khoảng trắng."""
    text = str(text).lower().replace('đ', 'd')
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', text).strip()


def canonical_object(name):
    """'Red Cube', 'red', 'khối đỏ' -> 'red_cube'; None nếu không nhận ra."""
    if name is None:
        return None
    n = normalize(name).replace('-', '_').replace(' ', '_')
    if n in OBJECTS:
        return n
    n = n.replace('_', ' ')
    for prefix in ('the ', 'khoi ', 'vat ', 'hop '):
        if n.startswith(prefix):
            n = n[len(prefix):]
    n = re.sub(r'\b(cube|block|box)\b', '', n).strip()
    for obj, aliases in _ALIASES_OBJ.items():
        if n in aliases:
            return obj
    return None


def canonical_zone(name):
    """'Zone B', 'zone_b', 'ô B', 'b' -> 'zone_b'; 'temp' -> TEMP; None nếu sai."""
    if name is None:
        return None
    n = normalize(name).replace('-', '_').replace(' ', '_')
    if n in ZONES or n == TEMP:
        return n
    if n in ('temp', 'temporary', 'temp_position', 'temporary_zone', 'free_position', 'vi_tri_tam'):
        return TEMP
    m = re.fullmatch(r'(?:zone|vung|o|khu|khu_vuc)?_?([abc])', n)
    if m:
        return 'zone_' + m.group(1)
    return None


def student_mapping(student_id):
    """Trả về (P, {zone: object}) theo 2 chữ số cuối MSSV."""
    digits = re.sub(r'\D', '', str(student_id))
    if len(digits) < 2:
        raise ValueError(f'student_id "{student_id}" phải có ít nhất 2 chữ số')
    p = int(digits[-2:]) % 6
    colors = STUDENT_TABLE[p]
    return p, {zone: f'{color}_cube' for zone, color in zip(ZONES, colors)}


def share_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('ur3_llm_control')
    except Exception:  # chạy test ngoài ROS
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yaml(path_or_name):
    path = path_or_name
    if not os.path.isabs(path):
        path = os.path.join(share_dir(), 'config', path_or_name)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def table_surface_z(cfg):
    t = cfg['table']
    return t['center'][2] + t['size'][2] / 2.0


def resting_z(cfg):
    """Độ cao tâm khối khi đặt trên bàn."""
    return table_surface_z(cfg) + cfg['block_size'] / 2.0


def skill_label(step):
    """{'skill':'place','object':'red_cube','zone':'zone_b'} -> 'place(red_cube, zone_b)'."""
    args = [step[k] for k in ('object', 'zone') if step.get(k)]
    return f"{step['skill']}({', '.join(args)})"
