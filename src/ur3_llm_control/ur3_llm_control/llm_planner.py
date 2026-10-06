"""LLM Planner: câu lệnh tự nhiên -> kế hoạch JSON {"plan": [...]} qua 9Router (API tương thích OpenAI).

Cấu hình bằng biến môi trường:
  LLM_BASE_URL  (mặc định http://localhost:20128/v1 - 9Router)
  LLM_API_KEY   (API key tạo trong dashboard 9Router)
  LLM_MODEL     (tên model/combo trong 9Router, ví dụ "cc/claude-sonnet-4-5" hoặc tên combo của bạn)
"""
import json
import os
import re

import requests

DEFAULT_BASE_URL = 'http://localhost:20128/v1'


class LLMError(RuntimeError):
    pass


def extract_json(text):
    """Lấy object JSON đầu tiên trong câu trả lời (chịu được ```json ...``` và chữ thừa)."""
    if text is None:
        raise LLMError('LLM trả về rỗng')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)   # model "reasoning"
    m = re.search(r'```(?:json)?\s*(.*?)```', text, flags=re.S)
    if m:
        text = m.group(1)
    start = text.find('{')
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find('{', start + 1)
    raise LLMError(f'không tìm thấy JSON hợp lệ trong câu trả lời: {text[:200]!r}')


def scene_summary(scene):
    """Mô tả cảnh camera thấy, gửi kèm cho LLM (không có toạ độ khớp)."""
    objs = {}
    for name, o in sorted(scene.get('objects', {}).items()):
        if o.get('zone'):
            objs[name] = f"in {o['zone']}"
        elif o.get('on_table', True):
            objs[name] = 'on the table (outside the zones)'
        else:
            objs[name] = 'not on the table'
    zones = {z: (o or 'empty') for z, o in sorted(scene.get('zones', {}).items())}
    return {'objects_detected_by_camera': objs, 'zones': zones}


def build_user_message(command, scene, student, held=None):
    info = {
        'current_scene': scene_summary(scene),
        'gripper_holding': held or 'nothing',
        'student_mapping': {
            'student_id': student['student_id'],
            'P': student['P'],
            'zone_targets': student['mapping'],
        },
    }
    return (f'CURRENT SCENE AND STUDENT MAPPING:\n{json.dumps(info, ensure_ascii=False, indent=1)}\n\n'
            f'USER COMMAND: {command}\n\nReturn only the JSON plan.')


class LLMPlanner:
    def __init__(self, system_prompt, base_url=None, api_key=None, model=None, timeout=90.0):
        self.system_prompt = system_prompt
        self.base_url = (base_url or os.environ.get('LLM_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')
        self.api_key = api_key if api_key is not None else os.environ.get('LLM_API_KEY', '')
        self.model = model or os.environ.get('LLM_MODEL', '')
        self.timeout = timeout

    def describe(self):
        return f'{self.base_url} model={self.model or "(chưa đặt LLM_MODEL)"}'

    def chat(self, messages):
        if not self.model:
            raise LLMError('chưa đặt biến môi trường LLM_MODEL (tên model trong 9Router)')
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        body = {'model': self.model, 'messages': messages, 'temperature': 0, 'stream': False}
        try:
            r = requests.post(f'{self.base_url}/chat/completions', json=body,
                              headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise LLMError(f'không kết nối được LLM tại {self.base_url}: {e}') from e
        if r.status_code != 200:
            raise LLMError(f'LLM trả HTTP {r.status_code}: {r.text[:300]}')
        try:
            return r.json()['choices'][0]['message']['content']
        except (ValueError, KeyError, IndexError) as e:
            raise LLMError(f'phản hồi LLM sai định dạng: {r.text[:300]}') from e

    def plan(self, command, scene, student, held=None, feedback=None, previous=None):
        """Gọi LLM, trả (dict JSON, raw text). feedback: lỗi validator của lần trước để LLM sửa."""
        messages = [{'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': build_user_message(command, scene, student, held)}]
        if feedback:
            messages.append({'role': 'assistant', 'content': previous or ''})
            messages.append({'role': 'user', 'content':
                             f'The plan was rejected by the validator: {feedback}\n'
                             'Fix it and return only the corrected JSON plan.'})
        raw = self.chat(messages)
        return extract_json(raw), raw
