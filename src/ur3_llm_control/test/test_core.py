"""Unit test không cần ROS chạy: python3 -m pytest test/ -q"""
import os

import numpy as np
import pytest
import yaml

from ur3_llm_control import common as C
from ur3_llm_control import vision as V
from ur3_llm_control.llm_planner import extract_json
from ur3_llm_control.task_validator import TaskValidator, normalize_plan

CFG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')


@pytest.fixture(scope='module')
def cfg():
    with open(os.path.join(CFG_DIR, 'scene.yaml'), encoding='utf-8') as f:
        return yaml.safe_load(f)


def scene_with(cfg, positions):
    """positions: {name: (x, y)} -> scene giống output của perception."""
    objs = {n: {'x': x, 'y': y, 'z': C.resting_z(cfg), 'yaw': 0.0, 'on_table': True,
                'zone': V.zone_of(cfg, x, y)} for n, (x, y) in positions.items()}
    zones = {z: next((n for n, o in objs.items() if o['zone'] == z), None) for z in cfg['zones']}
    return {'objects': objs, 'zones': zones}


@pytest.fixture
def scene(cfg):
    return scene_with(cfg, {'red_cube': (0.18, 0.20), 'yellow_cube': (0.18, 0.07),
                            'blue_cube': (0.30, 0.0), 'green_cube': (0.18, -0.07),
                            'purple_cube': (0.18, -0.20)})


# ---------------------------------------------------------------- MSSV
@pytest.mark.parametrize('sid,p,colors', [
    ('20210042', 0, ('red', 'yellow', 'blue')),
    ('20210043', 1, ('red', 'blue', 'yellow')),
    ('20210014', 2, ('yellow', 'red', 'blue')),
    ('20210027', 3, ('yellow', 'blue', 'red')),
    ('20210010', 4, ('blue', 'red', 'yellow')),
    ('20210099', 3, ('yellow', 'blue', 'red')),
    ('22020005', 5, ('blue', 'yellow', 'red')),
])
def test_student_mapping(sid, p, colors):
    got_p, mapping = C.student_mapping(sid)
    assert got_p == p
    assert [mapping[z] for z in C.ZONES] == [f'{c}_cube' for c in colors]


# ---------------------------------------------------------------- tên vật / zone
@pytest.mark.parametrize('text,obj', [
    ('red_cube', 'red_cube'), ('Red cube', 'red_cube'), ('khối màu đỏ', 'red_cube'),
    ('vàng', 'yellow_cube'), ('xanh dương', 'blue_cube'), ('xanh lá', 'green_cube'),
    ('tím', 'purple_cube'), ('orange', None), ('black_cube', None),
])
def test_canonical_object(text, obj):
    assert C.canonical_object(text) == obj


@pytest.mark.parametrize('text,zone', [
    ('zone_a', 'zone_a'), ('Zone B', 'zone_b'), ('vùng C', 'zone_c'), ('ô A', 'zone_a'),
    ('temporary_position', C.TEMP), ('zone_d', None), ('kitchen', None),
])
def test_canonical_zone(text, zone):
    assert C.canonical_zone(text) == zone


# ---------------------------------------------------------------- extract_json
def test_extract_json_variants():
    plan = {'plan': [{'skill': 'home'}]}
    assert extract_json('{"plan":[{"skill":"home"}]}') == plan
    assert extract_json('```json\n{"plan":[{"skill":"home"}]}\n```') == plan
    assert extract_json('<think>{"x":1}</think> Here: {"plan":[{"skill":"home"}]} done') == plan


# ---------------------------------------------------------------- validator
def test_valid_simple_plan(cfg, scene):
    r = TaskValidator(cfg).validate({'plan': [{'skill': 'pick', 'object': 'yellow_cube'},
                                              {'skill': 'place', 'object': 'yellow_cube', 'zone': 'zone_a'},
                                              {'skill': 'home'}]}, scene)
    assert r.ok, r.message


@pytest.mark.parametrize('plan,status', [
    ([{'skill': 'fly', 'object': 'red_cube'}], C.INVALID_SKILL),
    ([{'skill': 'pick', 'object': 'orange_cube'}], C.INVALID_OBJECT),
    ([{'skill': 'pick', 'object': 'red_cube'}, {'skill': 'place', 'object': 'red_cube', 'zone': 'zone_d'}],
     C.INVALID_ZONE),
    ([{'skill': 'place', 'object': 'red_cube', 'zone': 'zone_a'}], C.FAILED),            # chưa pick
    ([{'skill': 'pick', 'object': 'red_cube'}, {'skill': 'pick', 'object': 'green_cube'}], C.FAILED),
    ([{'skill': 'pick'}], C.FAILED),                                                       # thiếu tham số
    ([{'skill': 'pick', 'object': 'red_cube', 'joint_angles': [0, 1]}], C.FAILED),         # tham số lạ
    ([{'skill': 'pick', 'object': 'red_cube'}, {'skill': 'place', 'object': 'red_cube', 'zone': 'zone_b'}],
     C.ZONE_OCCUPIED),                                                                     # zone_b có blue
])
def test_rejections(cfg, scene, plan, status):
    r = TaskValidator(cfg).validate({'plan': plan}, scene)
    assert not r.ok
    assert r.status == status


def test_occupied_zone_sequence(cfg, scene):
    plan = [{'skill': 'home'}, {'skill': 'check_zone', 'zone': 'zone_b'},
            {'skill': 'find_free_position', 'object': 'blue_cube'},
            {'skill': 'pick', 'object': 'blue_cube'},
            {'skill': 'place', 'object': 'blue_cube', 'zone': 'temporary_position'},
            {'skill': 'pick', 'object': 'red_cube'},
            {'skill': 'place', 'object': 'red_cube', 'zone': 'zone_b'}, {'skill': 'home'}]
    r = TaskValidator(cfg).validate({'plan': plan}, scene)
    assert r.ok, r.message
    assert TaskValidator.expected_final(r.plan, scene)['zone_b'] == 'red_cube'


def test_object_not_seen(cfg, scene):
    del scene['objects']['purple_cube']
    r = TaskValidator(cfg).validate({'plan': [{'skill': 'pick', 'object': 'purple_cube'}]}, scene)
    assert r.status == C.INVALID_OBJECT


def test_normalize_names():
    plan, err = normalize_plan({'plan': [{'skill': 'Pick', 'object': 'khối đỏ'},
                                         {'skill': 'place', 'object': 'red', 'zone': 'vùng A'}]})
    assert err is None
    assert plan[1] == {'skill': 'place', 'object': 'red_cube', 'zone': 'zone_a'}


def test_temporary_only_for_place():
    _, err = normalize_plan({'plan': [{'skill': 'check_zone', 'zone': 'temporary_position'}]})
    assert err.status == C.INVALID_ZONE


# ---------------------------------------------------------------- vision
def test_render_detect_roundtrip(cfg):
    cam = V.CameraModel.from_config(cfg['camera'])
    blocks = {'red_cube': dict(color='red', x=0.18, y=0.2, z=0.018, yaw=0.3),
              'blue_cube': dict(color='blue', x=0.30, y=0.0, z=0.118, yaw=-0.2),   # đang được nhấc
              'purple_cube': dict(color='purple', x=0.30, y=-0.15, z=0.018, yaw=0.0)}
    img, depth = V.render(cam, cfg, blocks, np.random.default_rng(0))
    dets = {d['color']: d for d in V.detect(cam, cfg, img, depth)}
    assert set(dets) == {'red', 'blue', 'purple'}
    for name, b in blocks.items():
        d = dets[b['color']]
        assert abs(d['x'] - b['x']) < 0.005 and abs(d['y'] - b['y']) < 0.005
        assert abs(d['z'] - b['z']) < 0.01
    scene = V.build_scene(cfg, list(dets.values()), C.resting_z(cfg))
    assert scene['objects']['blue_cube']['on_table'] is False
    assert scene['zones'] == {'zone_a': None, 'zone_b': None, 'zone_c': 'purple_cube'}


def test_find_free_position(cfg, scene):
    x, y = V.find_free_position(cfg, scene['objects'])
    fs = cfg['free_space']
    for o in scene['objects'].values():
        assert np.hypot(x - o['x'], y - o['y']) >= fs['clearance'] - 1e-9
    assert V.zone_of(cfg, x, y) is None
    assert fs['min_radius'] <= np.hypot(x, y) <= fs['max_radius']
