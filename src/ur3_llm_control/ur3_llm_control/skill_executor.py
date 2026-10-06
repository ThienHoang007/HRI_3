"""Skill Executor: chạy kế hoạch đã được validator duyệt, từng skill một, in trạng thái.

Dừng ngay khi một skill không SUCCESS (không chạy tiếp các bước phụ thuộc).
"""
from dataclasses import dataclass, field

from . import common as C

LABEL_WIDTH = 23


def exec_line(label, status):
    """'pick(red_cube) ........ SUCCESS' (căn cột giống ví dụ trong đề)."""
    dots = LABEL_WIDTH - len(label)
    return f'{label} {"." * dots} {status}' if dots > 0 else f'{label} {status}'


def format_plan(plan):
    return '\n'.join(f'{i}. {C.skill_label(s)}' for i, s in enumerate(plan, 1))


@dataclass
class ExecutionReport:
    success: bool
    results: list = field(default_factory=list)   # [(step, SkillResult)]

    @property
    def failed(self):
        return next(((s, r) for s, r in self.results if not r.ok), None)


class SkillExecutor:
    def __init__(self, skills, printer=print, verbose=True):
        self.skills = skills
        self.print = printer
        self.verbose = verbose

    def execute(self, plan):
        report = ExecutionReport(True)
        for step in plan:
            label = C.skill_label(step)
            result = self.skills.run(step)
            report.results.append((step, result))
            line = exec_line(label, result.status)
            if self.verbose and result.message:
                line += f'   [{result.message}]'
            self.print(line)
            if not result.ok:
                report.success = False
                break
        return report
