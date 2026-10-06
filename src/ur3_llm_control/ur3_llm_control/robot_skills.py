"""Robot Skills: các skill mức cao, mọi chuyển động đều đi qua MoveIt 2.

LLM chỉ chọn skill + tham số. Ở đây mới tính toạ độ (từ camera) và để MoveIt lập quỹ đạo
(OMPL cho di chuyển tự do, Cartesian path cho đoạn hạ/nâng thẳng đứng), có kiểm tra
giới hạn khớp, tự va chạm và va chạm môi trường (bàn + các khối trong planning scene).

Mỗi skill trả về SkillResult(status, message) với status thuộc
SUCCESS / FAILED / INVALID_OBJECT / INVALID_ZONE / PLANNING_FAILED.
"""
import json
import math
import threading
import time
from dataclasses import dataclass

from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, Constraints,
                             JointConstraint, MotionPlanRequest, MoveItErrorCodes,
                             PlanningOptions, PlanningScene, RobotState)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Trigger

from . import common as C
from .vision import find_free_position

ARM_GROUP = 'ur_manipulator'
TCP_LINK = 'gripper_tcp'
ARM_JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
NAMED_POSES = {   # trùng với group_state trong SRDF
    'home': [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
}
GRIPPER_LINKS = ['gripper_base', 'finger_left', 'finger_right', 'gripper_tcp', 'wrist_3_link']

PLANNING_ERRORS = {
    MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.INVALID_MOTION_PLAN,
    MoveItErrorCodes.NO_IK_SOLUTION, MoveItErrorCodes.GOAL_IN_COLLISION,
    MoveItErrorCodes.START_STATE_IN_COLLISION, MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED,
    MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS, MoveItErrorCodes.TIMED_OUT,
    MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS,
    MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE,
}


@dataclass
class SkillResult:
    status: str
    message: str = ''

    @property
    def ok(self):
        return self.status == C.SUCCESS


class SkillError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.result = SkillResult(status, message)


def down_quaternion(yaw):
    """Hướng gripper chĩa thẳng xuống bàn, xoay yaw quanh trục z: Rz(yaw) * Rx(pi)."""
    return (math.cos(yaw / 2.0), math.sin(yaw / 2.0), 0.0, 0.0)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class RobotSkills:
    def __init__(self, node, cfg):
        self.node = node
        self.cfg = cfg
        self.log = node.get_logger()
        self.size = cfg['block_size']
        self.rest_z = C.resting_z(cfg)
        g, m = cfg['grasp'], cfg['motion']
        self.approach = g['approach_height']
        self.open_pos = g['open_position']
        self.close_pos = self.size / 2.0 - g['squeeze']
        self.release_gap = g['release_gap']
        self.vel = m['velocity_scaling']
        self.acc = m['acceleration_scaling']
        self.plan_time = m['planning_time']
        self.attempts = m['planning_attempts']
        self.cart_step = m['cartesian_step']

        self.scene = {'objects': {}, 'zones': {z: None for z in C.ZONES}}
        self.held = None
        self.temp_position = None
        self.joints = {}
        self._joint_lock = threading.Lock()

        node.create_subscription(JointState, '/joint_states', self._on_joints, 20)
        self.move_ac = ActionClient(node, MoveGroup, '/move_action')
        self.exec_ac = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
        self.grip_ac = ActionClient(node, GripperCommand, '/gripper_controller/gripper_cmd')
        self.ik_cli = node.create_client(GetPositionIK, '/compute_ik')
        self.cart_cli = node.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.scene_cli = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.detect_cli = node.create_client(Trigger, '/perception/detect')

    # ================================================================== hạ tầng
    def _on_joints(self, msg):
        with self._joint_lock:
            for n, p in zip(msg.name, msg.position):
                self.joints[n] = p

    def arm_joints(self):
        with self._joint_lock:
            return [self.joints.get(j, 0.0) for j in ARM_JOINTS]

    @staticmethod
    def _wait(future, timeout):
        ev = threading.Event()
        future.add_done_callback(lambda _f: ev.set())
        if not ev.wait(timeout):
            raise SkillError(C.FAILED, 'hết thời gian chờ phản hồi')
        return future.result()

    def wait_for_servers(self, timeout=60.0):
        """Chờ move_group, controller và perception sẵn sàng."""
        items = [('move_group /move_action', lambda t: self.move_ac.wait_for_server(timeout_sec=t)),
                 ('move_group /execute_trajectory', lambda t: self.exec_ac.wait_for_server(timeout_sec=t)),
                 ('gripper_controller', lambda t: self.grip_ac.wait_for_server(timeout_sec=t)),
                 ('/compute_ik', lambda t: self.ik_cli.wait_for_service(timeout_sec=t)),
                 ('/compute_cartesian_path', lambda t: self.cart_cli.wait_for_service(timeout_sec=t)),
                 ('/apply_planning_scene', lambda t: self.scene_cli.wait_for_service(timeout_sec=t)),
                 ('/perception/detect', lambda t: self.detect_cli.wait_for_service(timeout_sec=t))]
        for name, fn in items:
            if not fn(timeout):
                raise RuntimeError(f'không thấy {name} - đã chạy llm_robot.launch.py chưa?')
        t0 = time.monotonic()
        while not all(j in self.joints for j in ARM_JOINTS):
            if time.monotonic() - t0 > timeout:
                raise RuntimeError('không nhận được /joint_states')
            time.sleep(0.1)

    def _robot_state(self, joints=None):
        rs = RobotState()
        rs.is_diff = joints is None
        if joints is not None:
            rs.joint_state.name = list(ARM_JOINTS)
            rs.joint_state.position = list(joints)
        return rs

    # ------------------------------------------------------------------ MoveIt
    def _ik(self, xyz, yaw, seed):
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = ARM_GROUP
        r.robot_state = self._robot_state(seed)
        r.avoid_collisions = True
        r.ik_link_name = TCP_LINK
        r.timeout.nanosec = 200_000_000
        r.pose_stamped = self._pose(xyz, yaw)
        resp = self._wait(self.ik_cli.call_async(req), 5.0)
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        js = resp.solution.joint_state
        return [js.position[js.name.index(j)] for j in ARM_JOINTS]

    def _pose(self, xyz, yaw):
        ps = PoseStamped()
        ps.header.frame_id = 'world'
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = [float(v) for v in xyz]
        q = down_quaternion(yaw)
        ps.pose.orientation.x, ps.pose.orientation.y = q[0], q[1]
        ps.pose.orientation.z, ps.pose.orientation.w = q[2], q[3]
        return ps

    def best_ik(self, xyz, yaws):
        """Thử nhiều góc yaw (khối vuông đối xứng 90°), chọn nghiệm IK gần tư thế hiện tại nhất.
        Trả về (joints, yaw)."""
        cur = self.arm_joints()
        best = None
        for yaw in yaws:
            for seed in (cur, NAMED_POSES['home']):
                sol = self._ik(xyz, yaw, seed)
                if sol is None:
                    continue
                cost = sum(abs(wrap(a - b)) if i != 5 else abs(a - b)
                           for i, (a, b) in enumerate(zip(sol, cur)))
                cost += 0.5 * abs(sol[5])         # tránh xoắn cổ tay quá xa
                if best is None or cost < best[0]:
                    best = (cost, sol, yaw)
        if best is None:
            raise SkillError(C.PLANNING_FAILED,
                             f'không có nghiệm IK tới ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})')
        return best[1], best[2]

    def move_joints(self, target, what):
        """Lập quỹ đạo OMPL tới cấu hình khớp và thực thi (MoveGroup action)."""
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.start_state = self._robot_state()
        req.num_planning_attempts = self.attempts
        req.allowed_planning_time = float(self.plan_time)
        req.max_velocity_scaling_factor = self.vel
        req.max_acceleration_scaling_factor = self.acc
        c = Constraints()
        for name, pos in zip(ARM_JOINTS, target):
            jc = JointConstraint()
            jc.joint_name, jc.position = name, float(pos)
            jc.tolerance_above = jc.tolerance_below = 0.001
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        self._run_action(self.move_ac, goal, what, 60.0)

    def _run_action(self, client, goal, what, timeout):
        handle = self._wait(client.send_goal_async(goal), 10.0)
        if not handle.accepted:
            raise SkillError(C.FAILED, f'{what}: goal bị từ chối')
        res = self._wait(handle.get_result_async(), timeout).result
        code = res.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            return
        if code in PLANNING_ERRORS:
            raise SkillError(C.PLANNING_FAILED, f'{what}: MoveIt không lập được quỹ đạo (mã {code})')
        raise SkillError(C.FAILED, f'{what}: thực thi thất bại (mã {code})')

    def move_cartesian(self, xyz, yaw, what):
        """Đi thẳng tới điểm (hạ/nâng theo phương đứng) bằng Cartesian path, có kiểm tra va chạm."""
        req = GetCartesianPath.Request()
        req.header.frame_id = 'world'
        req.start_state = self._robot_state()
        req.group_name = ARM_GROUP
        req.link_name = TCP_LINK
        req.waypoints = [self._pose(xyz, yaw).pose]
        req.max_step = float(self.cart_step)
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = self.vel * 0.5
        req.max_acceleration_scaling_factor = self.acc * 0.5
        resp = self._wait(self.cart_cli.call_async(req), 10.0)
        if resp.error_code.val != MoveItErrorCodes.SUCCESS or resp.fraction < 0.99:
            raise SkillError(C.PLANNING_FAILED,
                             f'{what}: Cartesian path chỉ đạt {resp.fraction * 100:.0f}%')
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = resp.solution
        self._run_action(self.exec_ac, goal, what, 30.0)

    def gripper(self, position, what):
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 20.0
        handle = self._wait(self.grip_ac.send_goal_async(goal), 5.0)
        if not handle.accepted:
            raise SkillError(C.FAILED, f'{what}: gripper từ chối lệnh')
        self._wait(handle.get_result_async(), 15.0)
        if abs(self.joints.get('finger_joint', -1.0) - position) > 0.004:
            raise SkillError(C.FAILED, f'{what}: gripper không tới vị trí {position:.3f}')

    # ------------------------------------------------------------------ planning scene
    def _apply(self, scene_msg):
        resp = self._wait(self.scene_cli.call_async(ApplyPlanningScene.Request(scene=scene_msg)), 5.0)
        if not resp.success:
            raise SkillError(C.FAILED, 'không cập nhật được planning scene')

    @staticmethod
    def _box(oid, frame, xyz, size, yaw=0.0):
        co = CollisionObject()
        co.id = oid
        co.header.frame_id = frame
        prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(s) for s in size])
        p = Pose()
        p.position.x, p.position.y, p.position.z = [float(v) for v in xyz]
        p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        co.primitives.append(prim)
        co.primitive_poses.append(p)
        co.operation = CollisionObject.ADD
        return co

    def setup_scene(self):
        t = self.cfg['table']
        ps = PlanningScene(is_diff=True)
        ps.world.collision_objects.append(self._box('table', 'world', t['center'], t['size']))
        self._apply(ps)

    def _sync_scene(self):
        """Đưa các khối camera thấy (trừ khối đang cầm) vào planning scene làm vật cản."""
        ps = PlanningScene(is_diff=True)
        for name in C.OBJECTS:
            if name == self.held:
                continue
            o = self.scene['objects'].get(name)
            if o:
                ps.world.collision_objects.append(
                    self._box(name, 'world', (o['x'], o['y'], o['z']), [self.size] * 3, o.get('yaw', 0.0)))
            else:
                ps.world.collision_objects.append(CollisionObject(id=name, operation=CollisionObject.REMOVE))
        self._apply(ps)

    def _attach(self, name):
        # Cùng id với object trong world -> MoveIt tự gỡ khỏi world và gắn vào tay
        ps = PlanningScene(is_diff=True)
        aco = AttachedCollisionObject()
        aco.link_name = TCP_LINK
        aco.object = self._box(name, TCP_LINK, (0, 0, 0), [self.size] * 3)
        aco.touch_links = GRIPPER_LINKS
        ps.robot_state.attached_collision_objects.append(aco)
        ps.robot_state.is_diff = True
        self._apply(ps)

    def _detach(self, name, xyz=None, yaw=0.0):
        """Gỡ khối khỏi tay. xyz=None: chưa đưa lại vào world (để tay rút lên không bị coi là
        va chạm với khối vừa thả); perceive() sau đó sẽ thêm lại theo vị trí camera thấy."""
        ps = PlanningScene(is_diff=True)
        aco = AttachedCollisionObject()
        aco.link_name = TCP_LINK
        aco.object.id = name
        aco.object.operation = CollisionObject.REMOVE
        ps.robot_state.attached_collision_objects.append(aco)
        ps.robot_state.is_diff = True
        if xyz is not None:
            ps.world.collision_objects.append(self._box(name, 'world', xyz, [self.size] * 3, yaw))
        self._apply(ps)

    # ------------------------------------------------------------------ camera
    def perceive(self):
        resp = self._wait(self.detect_cli.call_async(Trigger.Request()), 10.0)
        if not resp.success:
            raise SkillError(C.FAILED, f'camera: {resp.message}')
        self.scene = json.loads(resp.message)
        if self.held:     # khối đang cầm nằm trong tay, không phải trên bàn
            self.scene['objects'].pop(self.held, None)
            self.scene['zones'] = {z: (None if o == self.held else o) for z, o in self.scene['zones'].items()}
        self._sync_scene()
        return self.scene

    def _seen(self, obj):
        o = self.scene['objects'].get(obj)
        if o is None:
            raise SkillError(C.INVALID_OBJECT, f'camera không thấy {obj}')
        return o

    def _target_xy(self, zone, obj=None):
        if zone == C.TEMP:
            if self.temp_position is None:
                self.temp_position = find_free_position(self.cfg, self.scene['objects'], ignore=[obj])
            if self.temp_position is None:
                raise SkillError(C.PLANNING_FAILED, 'không tìm được chỗ trống trên bàn')
            return self.temp_position
        if zone not in self.cfg['zones']:
            raise SkillError(C.INVALID_ZONE, f'zone "{zone}" không tồn tại')
        return tuple(self.cfg['zones'][zone]['center'])

    # ================================================================== SKILLS
    def run(self, step):
        """Chạy 1 bước kế hoạch, luôn trả SkillResult (không ném lỗi)."""
        fn = getattr(self, 'skill_' + step['skill'], None)
        if fn is None:
            return SkillResult(C.FAILED, f'skill {step["skill"]} chưa được cài đặt')
        kwargs = {k: step[k] for k in ('object', 'zone') if k in step}
        try:
            return fn(**kwargs)
        except SkillError as e:
            return e.result
        except Exception as e:  # noqa: BLE001 - báo lỗi thay vì làm sập node
            self.log.error(f'lỗi khi chạy {step}: {e!r}')
            return SkillResult(C.FAILED, str(e))

    def skill_home(self):
        self.move_joints(NAMED_POSES['home'], 'home')
        return SkillResult(C.SUCCESS)

    def skill_open_gripper(self):
        if self.held:
            raise SkillError(C.FAILED, f'đang cầm {self.held}; dùng place để thả')
        self.gripper(self.open_pos, 'open_gripper')
        return SkillResult(C.SUCCESS)

    def skill_close_gripper(self):
        if self.held:
            return SkillResult(C.SUCCESS, f'đang kẹp {self.held}')
        self.gripper(0.0, 'close_gripper')
        return SkillResult(C.SUCCESS)

    def skill_detect_objects(self):
        s = self.perceive()
        return SkillResult(C.SUCCESS, f"thấy {len(s['objects'])} khối: {', '.join(sorted(s['objects']))}")

    def skill_find_object(self, object):  # noqa: A002 - tên tham số theo JSON
        self.perceive()
        o = self._seen(object)
        where = o['zone'] or f"({o['x']:.3f}, {o['y']:.3f})"
        return SkillResult(C.SUCCESS, f'{object} ở {where}')

    def skill_check_zone(self, zone):
        if zone not in self.cfg['zones']:
            raise SkillError(C.INVALID_ZONE, f'zone "{zone}" không tồn tại')
        occ = self.perceive()['zones'].get(zone)
        return SkillResult(C.SUCCESS, f'{zone} đang có {occ}' if occ else f'{zone} trống')

    def skill_find_free_position(self, object=None):  # noqa: A002
        self.perceive()
        self.temp_position = find_free_position(self.cfg, self.scene['objects'], ignore=[object] if object else [])
        if self.temp_position is None:
            raise SkillError(C.PLANNING_FAILED, 'không còn chỗ trống trên bàn')
        x, y = self.temp_position
        return SkillResult(C.SUCCESS, f'temporary_position = ({x:.3f}, {y:.3f})')

    def _move_above_xy(self, x, y, z_base, yaws, what):
        joints, yaw = self.best_ik((x, y, z_base + self.approach), yaws)
        self.move_joints(joints, what)
        return yaw

    def skill_move_above(self, object=None, zone=None):  # noqa: A002
        if object:
            self.perceive()
            o = self._seen(object)
            yaws = [o.get('yaw', 0.0) + k * math.pi / 2 for k in range(-2, 2)]
            self._move_above_xy(o['x'], o['y'], o['z'], yaws, f'move_above({object})')
        else:
            x, y = self._target_xy(zone)
            self._move_above_xy(x, y, self.rest_z, [k * math.pi / 2 for k in range(-2, 2)], f'move_above({zone})')
        return SkillResult(C.SUCCESS)

    def skill_move_to_zone(self, zone):
        return self.skill_move_above(zone=zone)

    def skill_pick(self, object):  # noqa: A002
        if object not in C.OBJECTS:
            raise SkillError(C.INVALID_OBJECT, f'"{object}" không phải vật hợp lệ')
        if self.held:
            raise SkillError(C.FAILED, f'đang cầm {self.held}')
        self.perceive()
        o = self._seen(object)
        if not o.get('on_table', True):
            raise SkillError(C.FAILED, f'{object} không nằm trên bàn')
        what = f'pick({object})'
        self.gripper(self.open_pos, what)
        yaws = [o.get('yaw', 0.0) + k * math.pi / 2 for k in range(-2, 2)]
        yaw = self._move_above_xy(o['x'], o['y'], o['z'], yaws, what)
        grasp = (o['x'], o['y'], o['z'])
        # Ngón mở 8 cm > cạnh khối 4 cm nên hạ xuống không chạm khối; kẹp xong mới gắn khối vào tay
        self.move_cartesian(grasp, yaw, what + ' hạ xuống')
        self.gripper(self.close_pos, what + ' kẹp')
        self.held = object
        self._attach(object)
        self.move_cartesian((o['x'], o['y'], o['z'] + self.approach), yaw, what + ' nhấc lên')
        # Kiểm tra bằng camera: khối phải rời khỏi bàn (đi theo gripper)
        s = self._raw_scene()
        lifted = s['objects'].get(object)
        if lifted is None or lifted['on_table']:
            self.held = None
            self._detach(object, grasp, o.get('yaw', 0.0))
            raise SkillError(C.FAILED, f'kẹp trượt: camera vẫn thấy {object} trên bàn ({lifted})')
        return SkillResult(C.SUCCESS)

    def _raw_scene(self):
        resp = self._wait(self.detect_cli.call_async(Trigger.Request()), 10.0)
        if not resp.success:
            raise SkillError(C.FAILED, f'camera: {resp.message}')
        return json.loads(resp.message)

    def skill_place(self, object, zone):  # noqa: A002
        if object not in C.OBJECTS:
            raise SkillError(C.INVALID_OBJECT, f'"{object}" không phải vật hợp lệ')
        if zone != C.TEMP and zone not in self.cfg['zones']:
            raise SkillError(C.INVALID_ZONE, f'zone "{zone}" không tồn tại')
        if self.held != object:
            raise SkillError(C.FAILED, f'không cầm {object} (đang cầm: {self.held or "không"})')
        self.perceive()
        what = f'place({object}, {zone})'
        if zone != C.TEMP:
            occ = self.scene['zones'].get(zone)
            if occ:
                raise SkillError(C.FAILED, f'{zone} đang bị {occ} chiếm')
        x, y = self._target_xy(zone, object)
        z = self.rest_z + self.release_gap
        yaws = [k * math.pi / 2 for k in range(-2, 2)]
        yaw = self._move_above_xy(x, y, z, yaws, what)
        self.move_cartesian((x, y, z), yaw, what + ' hạ xuống')
        self.gripper(self.open_pos, what + ' thả')
        self.held = None
        self._detach(object)
        self.move_cartesian((x, y, z + self.approach), yaw, what + ' rút lên')
        s = self.perceive()
        o = s['objects'].get(object)
        if zone == C.TEMP:
            self.temp_position = None
            ok = o is not None and o['on_table'] and math.hypot(o['x'] - x, o['y'] - y) < 0.02
        else:
            ok = o is not None and o['zone'] == zone
        if not ok:
            raise SkillError(C.FAILED, f'camera không thấy {object} ở {zone} sau khi thả')
        return SkillResult(C.SUCCESS)
