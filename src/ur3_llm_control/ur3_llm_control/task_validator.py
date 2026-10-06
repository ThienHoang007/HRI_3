"""Plan Validator: kiểm tra kế hoạch JSON của LLM trước khi thực thi.

Kiểm tra:
  1. Định dạng: {"plan": [ {"skill": ..., ...}, ... ]}
  2. Skill hợp lệ, đủ tham số; object/zone nằm trong danh sách cho phép.
  3. Mô phỏng trạng thái (dựa trên cảnh camera thấy) theo từng bước:
     - pick: tay đang rỗng, vật được camera phát hiện và đang nằm trên bàn.
     - place: đang cầm đúng vật đó; zone đích phải TRỐNG tại thời điểm đặt
       (nếu bận -> phải dời vật chắn sang temporary_position trước).
     - temporary_position: phải tìm được chỗ trống trên bàn.
Không sửa kế hoạch: nếu sai thì trả lỗi để LLM lập lại.
"""
import copy
from dataclasses import dataclass, field

from . import common as C
from .vision import find_free_position


@dataclass
class ValidationResult:
    ok: bool
    status: str = C.SUCCESS
    message: str = ''
    plan: list = field(default_factory=list)   # kế hoạch đã chuẩn hoá tên


def _fail(status, msg, plan=None):
    return ValidationResult(False, status, msg, plan or [])


def normalize_plan(data):
    """Chuẩn hoá tên skill/object/zone. Trả (plan, None) hoặc (None, ValidationResult lỗi)."""
    if isinstance(data, dict):
        data = data.get('plan')
    if not isinstance(data, list):
        return None, _fail(C.FAILED, 'JSON phải có dạng {"plan": [...]}')
    if not data:
        return None, _fail(C.FAILED, 'kế hoạch rỗng')
    plan = []
    for i, raw in enumerate(data, 1):
        if not isinstance(raw, dict) or 'skill' not in raw:
            return None, _fail(C.FAILED, f'bước {i}: thiếu trường "skill"')
        skill = C.normalize(raw['skill']).replace(' ', '_')
        if skill not in C.SKILLS:
            return None, _fail(C.INVALID_SKILL, f'bước {i}: skill "{raw["skill"]}" không tồn tại')
        required, optional = C.SKILLS[skill]
        step = {'skill': skill}
        for key in required + optional:
            if key not in raw or raw[key] in (None, ''):
                if key in required:
                    return None, _fail(C.FAILED, f'bước {i}: {skill} thiếu tham số "{key}"')
                continue
            if key == 'object':
                val = C.canonical_object(raw[key])
                if val is None:
                    return None, _fail(C.INVALID_OBJECT, f'bước {i}: object "{raw[key]}" không hợp lệ')
            else:
                val = C.canonical_zone(raw[key])
                if val is None or (val == C.TEMP and skill != 'place'):
                    return None, _fail(C.INVALID_ZONE, f'bước {i}: zone "{raw[key]}" không hợp lệ')
            step[key] = val
        extra = set(raw) - {'skill'} - set(required) - set(optional)
        if extra:
            return None, _fail(C.FAILED, f'bước {i}: {skill} có tham số lạ {sorted(extra)}')
        if skill == 'move_above' and not ('object' in step) ^ ('zone' in step):
            return None, _fail(C.FAILED, f'bước {i}: move_above cần đúng 1 trong "object" hoặc "zone"')
        plan.append(step)
    return plan, None


class TaskValidator:
    def __init__(self, cfg):
        self.cfg = cfg

    def validate(self, data, scene, held=None):
        """scene: kết quả camera {'objects': {name: {...}}, 'zones': {zone: name|None}}.
        held: tên vật đang cầm (None nếu tay rỗng)."""
        plan, err = normalize_plan(data)
        if err:
            return err

        objs = copy.deepcopy(scene.get('objects', {}))
        holding = held
        for i, step in enumerate(plan, 1):
            s, obj, zone = step['skill'], step.get('object'), step.get('zone')
            where = f'bước {i} {C.skill_label(step)}'

            if obj and obj not in objs and obj != holding:
                return _fail(C.INVALID_OBJECT, f'{where}: camera không thấy {obj} trên bàn', plan)

            if s == 'pick':
                if holding:
                    return _fail(C.FAILED, f'{where}: tay đang cầm {holding}, phải place trước', plan)
                if obj == held and obj not in objs:
                    return _fail(C.FAILED, f'{where}: {obj} đang được cầm sẵn', plan)
                if not objs[obj].get('on_table', True):
                    return _fail(C.FAILED, f'{where}: {obj} không nằm trên bàn', plan)
                holding = obj
                objs.pop(obj)

            elif s == 'place':
                if holding != obj:
                    return _fail(C.FAILED, f'{where}: phải pick({obj}) trước khi place', plan)
                if zone == C.TEMP:
                    spot = find_free_position(self.cfg, objs)
                    if spot is None:
                        return _fail(C.PLANNING_FAILED, f'{where}: không còn chỗ trống trên bàn', plan)
                    objs[obj] = {'x': spot[0], 'y': spot[1], 'on_table': True, 'zone': None}
                else:
                    blocker = next((n for n, o in objs.items() if o.get('zone') == zone), None)
                    if blocker:
                        return _fail(C.ZONE_OCCUPIED,
                                     f'{where}: {zone} đang có {blocker}; phải check_zone, find_free_position, '
                                     f'pick({blocker}), place({blocker}, temporary_position) trước', plan)
                    cx, cy = self.cfg['zones'][zone]['center']
                    objs[obj] = {'x': cx, 'y': cy, 'on_table': True, 'zone': zone}
                holding = None

            elif s == 'move_above' and obj and obj == holding:
                return _fail(C.FAILED, f'{where}: không thể move_above vật đang cầm', plan)

        return ValidationResult(True, C.SUCCESS, 'OK', plan)

    @staticmethod
    def expected_final(plan, scene):
        """Trạng thái zone dự kiến sau khi chạy (để in ra)."""
        zones = dict(scene.get('zones', {}))
        for step in plan:
            if step['skill'] == 'pick':
                for z, o in zones.items():
                    if o == step['object']:
                        zones[z] = None
            elif step['skill'] == 'place' and step['zone'] in zones:
                zones[step['zone']] = step['object']
        return zones
