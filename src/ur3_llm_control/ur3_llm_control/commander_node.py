"""llm_commander: giao diện terminal.

Natural Language Command -> LLM Planner -> JSON Plan -> Plan Validator -> Skill Executor -> MoveIt 2 -> UR3 + Gripper

Chạy:  ros2 run ur3_llm_control llm_commander                      (nhập lệnh liên tục)
       ros2 run ur3_llm_control llm_commander "Put the red cube in zone B."   (chạy 1 lệnh rồi thoát)
       ros2 run ur3_llm_control llm_commander --plan '{"plan":[...]}'           (bỏ qua LLM, test validator/executor)
"""
import json
import os
import sys
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from . import common as C
from .llm_planner import LLMError, LLMPlanner, scene_summary
from .robot_skills import RobotSkills
from .skill_executor import SkillExecutor, format_plan
from .task_validator import TaskValidator

BOLD, GREEN, RED, YELLOW, RESET = '\033[1m', '\033[92m', '\033[91m', '\033[93m', '\033[0m'
if not sys.stdout.isatty():
    BOLD = GREEN = RED = YELLOW = RESET = ''


class Commander(Node):
    def __init__(self):
        super().__init__('llm_commander')
        self.declare_parameter('scene_file', 'scene.yaml')
        self.declare_parameter('student_file', 'student_config.yaml')
        self.declare_parameter('prompt_file', '')
        self.declare_parameter('max_llm_attempts', 3)
        self.cfg = C.load_yaml(self.get_parameter('scene_file').value)
        st = C.load_yaml(self.get_parameter('student_file').value)
        p, mapping = C.student_mapping(st['student_id'])
        self.student = {'student_name': st.get('student_name', ''), 'student_id': str(st['student_id']),
                        'P': p, 'mapping': mapping}

        prompt_file = self.get_parameter('prompt_file').value or os.path.join(
            C.share_dir(), 'prompts', 'system_prompt.md')
        with open(prompt_file, encoding='utf-8') as f:
            self.planner = LLMPlanner(f.read())
        self.max_attempts = int(self.get_parameter('max_llm_attempts').value)

        self.skills = RobotSkills(self, self.cfg)
        self.validator = TaskValidator(self.cfg)
        self.skill_executor = SkillExecutor(self.skills)
        self.reset_cli = self.create_client(Trigger, '/world_sim/reset')

    # ------------------------------------------------------------------
    def startup(self):
        print('Đang chờ MoveIt, controller và camera ...', flush=True)
        self.skills.wait_for_servers()
        self.skills.setup_scene()
        self.skills.perceive()
        st = self.student
        print(f"{BOLD}STUDENT:{RESET} {st['student_name']} - {st['student_id']}  ->  P = {st['P']}")
        print('  ' + ', '.join(f"{z[-1].upper()} = {o.split('_')[0].capitalize()}"
                               for z, o in st['mapping'].items()))
        print(f'{BOLD}LLM:{RESET} {self.planner.describe()}')
        self.print_scene()

    def print_scene(self):
        s = scene_summary(self.skills.scene)
        print(f'{BOLD}CAMERA:{RESET}')
        for name, where in s['objects_detected_by_camera'].items():
            print(f'  {name:<12} {where}')
        print('  ' + ' | '.join(f'{z}: {o}' for z, o in s['zones'].items()))

    def get_plan(self, command):
        """Hỏi LLM, kiểm tra bằng validator; nếu sai thì gửi lỗi lại cho LLM sửa (tối đa N lần)."""
        feedback, raw, last = None, None, None
        for attempt in range(1, self.max_attempts + 1):
            data, raw = self.planner.plan(command, self.skills.scene, self.student,
                                          held=self.skills.held, feedback=feedback, previous=raw)
            if isinstance(data, dict) and not data.get('plan') and data.get('error'):
                return None, f"LLM từ chối: {data['error']}"
            result = self.validator.validate(data, self.skills.scene, self.skills.held)
            if result.ok:
                return result.plan, None
            last = f'{result.status}: {result.message}'
            print(f'{YELLOW}VALIDATOR (lần {attempt}): REJECTED - {last}{RESET}')
            print(f'  plan: {json.dumps(data, ensure_ascii=False)}')
            feedback = last
        return None, last

    def run_command(self, command, plan_json=None):
        print(f'\n{BOLD}USER COMMAND:{RESET}\n{command}')
        try:
            self.skills.perceive()      # luôn lập kế hoạch dựa trên ảnh camera mới nhất
        except Exception as e:  # noqa: BLE001
            print(f'{RED}Camera lỗi: {e}{RESET}')
            return False
        if plan_json is not None:
            result = self.validator.validate(plan_json, self.skills.scene, self.skills.held)
            plan, err = (result.plan, None) if result.ok else (None, f'{result.status}: {result.message}')
        else:
            try:
                plan, err = self.get_plan(command)
            except LLMError as e:
                print(f'{RED}LLM lỗi: {e}{RESET}')
                return False
        if plan is None:
            print(f'{RED}PLAN REJECTED: {err}{RESET}\n{RED}{BOLD}TASK FAILED{RESET}')
            return False

        print(f'{BOLD}LLM PLAN:{RESET}\n{format_plan(plan)}')
        print(f'{BOLD}VALIDATION:{RESET} OK')
        print(f'{BOLD}EXECUTION:{RESET}')
        report = self.skill_executor.execute(plan)
        if report.success:
            print(f'{GREEN}{BOLD}TASK SUCCESS{RESET}')
        else:
            step, res = report.failed
            print(f'{RED}{BOLD}TASK FAILED{RESET} tại {C.skill_label(step)}: {res.status} {res.message}')
        return report.success

    def reset_world(self):
        if not self.reset_cli.wait_for_service(timeout_sec=2.0):
            print('không có /world_sim/reset')
            return
        ev = threading.Event()
        self.reset_cli.call_async(Trigger.Request()).add_done_callback(lambda _f: ev.set())
        ev.wait(5.0)
        self.skills.held = None
        self.skills.skill_home()
        self.skills.perceive()
        self.print_scene()


HELP = """Nhập lệnh tiếng Việt hoặc tiếng Anh, ví dụ:
  Put the red cube in zone B.
  Đưa khối màu đỏ vào vùng B
  Hãy lấy khối màu vàng và đặt nó vào ô A
  Move the blue cube to zone C
  Arrange all objects according to my student ID
Lệnh đặc biệt: :scene (xem camera), :reset (đặt lại các khối), :plan {json} (chạy plan tự viết), :quit"""


def main():
    args = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    rclpy.init()
    node = Commander()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    code = 0
    try:
        node.startup()
        if args and args[0] == '--plan':
            code = 0 if node.run_command('(manual plan)', json.loads(args[1])) else 1
        elif args:
            code = 0 if node.run_command(' '.join(args)) else 1
        else:
            print(HELP)
            while rclpy.ok():
                try:
                    line = input(f'\n{BOLD}>>> {RESET}').strip()
                except EOFError:
                    break
                if not line:
                    continue
                if line in (':quit', ':q', 'exit', 'quit'):
                    break
                if line == ':scene':
                    node.skills.perceive()
                    node.print_scene()
                elif line == ':reset':
                    node.reset_world()
                elif line.startswith(':plan'):
                    try:
                        node.run_command('(manual plan)', json.loads(line[5:]))
                    except json.JSONDecodeError as e:
                        print(f'JSON sai: {e}')
                elif line in (':help', 'help'):
                    print(HELP)
                else:
                    node.run_command(line)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f'{RED}{e}{RESET}')
        code = 1
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
