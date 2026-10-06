# ur3_llm_control — Điều khiển UR3 + Gripper bằng ngôn ngữ tự nhiên (Bài 02 + Bài 03)

```
Natural Language Command → LLM Planner → JSON Plan → Plan Validator → Skill Executor → MoveIt 2 → UR3 + Gripper
                                   ↑                        ↑               ↑
                          scene từ camera (RGB-D) ──────────┴───────────────┘
```

LLM **chỉ chọn skill, tham số và thứ tự**. Nó không sinh góc khớp, quỹ đạo hay toạ độ. Toàn bộ chuyển động do MoveIt 2 lập kế hoạch: IK + OMPL + Cartesian path, có kiểm tra giới hạn khớp và va chạm với bàn và các khối.

## Thành phần

| Node / file | Vai trò |
|---|---|
| `launch/llm_robot.launch.py` | UR3 + gripper (ros2_control fake hardware), MoveIt 2 `move_group`, RViz, `world_sim`, `perception` |
| `world_sim` | "Thế giới thật" mô phỏng: 5 khối (red, yellow, blue, green, purple). Vật lý kẹp: khối chỉ đi theo tay khi nằm giữa 2 ngón và ngón khép lại; khi ngón mở thì khối rơi xuống bàn. **Không dịch chuyển tức thời (teleport) khối.** Node này cũng phát camera RGB-D (`/camera/color/image_raw`, `/camera/depth/image_raw`, `/camera/camera_info`) đặt trên cao nhìn xuống bàn. |
| `perception` | OpenCV: tách màu theo HSV, lấy độ cao từ ảnh depth, góc xoay bằng `minAreaRect`. Từ đó suy ra vị trí khối và trạng thái từng zone. Service `/perception/detect`, topic `/perception/scene` và ảnh debug `/perception/debug_image`. |
| `llm_commander` | Terminal: nhận lệnh → LLM (9Router) → validator → executor, in kết quả theo đúng định dạng đề bài |
| `task_validator.py` | Kiểm tra skill, object, zone và tham số. Mô phỏng từng bước: tay rỗng/đang cầm, zone bận (ZONE_OCCUPIED), chỗ trống tạm. Nếu sai thì gửi lỗi lại cho LLM sửa, tối đa 3 lần. |
| `robot_skills.py` | Các skill dùng MoveIt 2 (`/move_action`, `/compute_ik`, `/compute_cartesian_path`, `/execute_trajectory`, `/apply_planning_scene`) và gripper (`/gripper_controller/gripper_cmd`) |
| `prompts/system_prompt.md` | System prompt của LLM |
| `config/scene.yaml` | Bàn, khối, zone, camera, tham số gắp |
| `config/student_config.yaml` | **Họ tên và MSSV, cần sửa thành thông tin của bạn** |

### Skill

Mỗi skill trả về `SUCCESS / FAILED / INVALID_OBJECT / INVALID_ZONE / PLANNING_FAILED`.

| Skill | Ý nghĩa |
|---|---|
| `home` | Về tư thế home |
| `pick(object)` | Mở kẹp → lên trên khối → hạ thẳng → kẹp → nhấc. Camera kiểm tra khối đã rời bàn. |
| `place(object, zone)` | Đến trên zone (hoặc `temporary_position`) → hạ → mở kẹp → rút lên. Camera kiểm tra khối nằm trong zone. |
| `check_zone(zone)` | Camera xem zone trống hay bận |
| `find_free_position(object)` | Tìm chỗ trống trên bàn để đặt tạm (xa các khối khác và các zone, robot với tới được) |
| `detect_objects`, `find_object(object)`, `move_above(object|zone)`, `move_to_zone(zone)`, `open_gripper`, `close_gripper` | Skill bổ sung |

Định dạng plan: `{"plan":[{"skill":"pick","object":"red_cube"},{"skill":"place","object":"red_cube","zone":"zone_b"},{"skill":"home"}]}`

### Ánh xạ theo MSSV

`P = (2 chữ số cuối MSSV) mod 6`, cho màu của zone A, B, C:

| P | A | B | C |
|---|---|---|---|
| 0 | Red | Yellow | Blue |
| 1 | Red | Blue | Yellow |
| 2 | Yellow | Red | Blue |
| 3 | Yellow | Blue | Red |
| 4 | Blue | Red | Yellow |
| 5 | Blue | Yellow | Red |

Lệnh `Arrange all objects according to my student ID` sẽ đặt 3 khối tương ứng vào 3 zone. Nếu zone đang bị chiếm thì robot dời khối chắn ra chỗ trống trước.

## Cài đặt và build

Yêu cầu: ROS 2 Humble, MoveIt 2, ros2_control, ur_description, OpenCV, numpy, requests.

```bash
sudo apt install ros-humble-moveit ros-humble-ur-description ros-humble-ros2-control ros-humble-ros2-controllers python3-opencv python3-requests
```

```bash
cd ~/Documents/HRI/4/ur3_llm_ws && source /opt/ros/humble/setup.bash && colcon build --symlink-install
```

## Cấu hình

1. **MSSV:** sửa `config/student_config.yaml` (`student_name`, `student_id`). Với `--symlink-install` thì không cần build lại.
2. **LLM qua 9Router:** chạy 9Router (mặc định ở `http://localhost:20128`), rồi đặt biến môi trường. `LLM_MODEL` là tên model/combo trong dashboard 9Router, ví dụ `cc/claude-sonnet-4-5` hoặc `gemini/gemini-2.5-flash`.

```bash
export LLM_BASE_URL=http://localhost:20128/v1 LLM_API_KEY=<key_9router> LLM_MODEL=<ten_model>
```

Dùng được mọi endpoint tương thích OpenAI (`/v1/chat/completions`) khác bằng cách đổi 3 biến trên.

## Chạy

Terminal 1: robot, MoveIt, RViz, camera, mô phỏng.

```bash
source install/setup.bash && ros2 launch ur3_llm_control llm_robot.launch.py
```

Terminal 2: giao diện nhập lệnh (cần `export LLM_...` như trên).

```bash
source install/setup.bash && ros2 run ur3_llm_control llm_commander
```

Ví dụ lệnh, gõ tiếng Việt hay tiếng Anh đều được:

```
Put the red cube in zone B.
Đưa khối màu đỏ vào vùng B
Hãy lấy khối màu vàng và đặt nó vào ô A
Move the blue cube to zone C
Arrange all objects according to my student ID
Sắp xếp các khối theo mã số sinh viên của tôi
```

Lệnh đặc biệt trong terminal:

- `:scene`: in những gì camera thấy
- `:reset`: đặt lại các khối như ban đầu
- `:plan {json}`: chạy một plan tự viết, bỏ qua LLM, để test validator/executor
- `:quit`: thoát

Chạy 1 lệnh rồi thoát:

```bash
ros2 run ur3_llm_control llm_commander "Put the red cube in zone B."
```

Trong RViz có robot, planning scene (bàn và khối là vật cản) và marker `/world_markers` (vị trí thật của khối). Ảnh camera kèm kết quả nhận dạng nằm ở panel `/perception/debug_image`.

## Demo zone bị chiếm (Bài 03)

Cảnh ban đầu trong `scene.yaml`: **blue_cube nằm sẵn trong zone_b**. Lệnh `Đưa khối màu đỏ vào vùng B` cho kết quả (đã chạy thử):

```
USER COMMAND:
Đưa khối màu đỏ vào vùng B
LLM PLAN:
1. home()
2. check_zone(zone_b)
3. find_free_position(blue_cube)
4. pick(blue_cube)
5. place(blue_cube, temporary_position)
6. pick(red_cube)
7. place(red_cube, zone_b)
8. home()
VALIDATION: OK
EXECUTION:
home() ................. SUCCESS
check_zone(zone_b) ..... SUCCESS   [zone_b đang có blue_cube]
find_free_position(blue_cube) SUCCESS   [temporary_position = (0.090, 0.260)]
pick(blue_cube) ........ SUCCESS
place(blue_cube, temporary_position) SUCCESS
pick(red_cube) ......... SUCCESS
place(red_cube, zone_b) SUCCESS
home() ................. SUCCESS
TASK SUCCESS
```

Nếu LLM bỏ qua bước dời khối chắn, validator sẽ từ chối với `ZONE_OCCUPIED` và gửi lỗi lại để LLM lập kế hoạch lại.

Có thể đổi vị trí ban đầu của khối (trường `initial`) trong `config/scene.yaml`.

## Test

```bash
cd src/ur3_llm_control && python3 -m pytest test/ -q
```

Gồm: ánh xạ MSSV, chuẩn hoá tên tiếng Việt/Anh, tách JSON từ phản hồi LLM, validator (skill/object/zone sai, chưa pick đã place, zone bận, tham số lạ như `joint_angles`), và render → detect của camera.
