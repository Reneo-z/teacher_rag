from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teacher_rag.skill_agent import answer_question


CASES = [
    {
        "question": "怎么建巡检任务？",
        "module": "任务管理",
        "expect": "skill_answer",
    },
    {
        "question": "创建巡检任务时，如何指派执行人？",
        "module": "任务管理",
        "expect": "skill_answer",
    },
    {
        "question": "如何修改个人头像？",
        "module": "个人中心",
        "expect": "no_skill_match",
    },
]


def main() -> int:
    skills_root = ROOT / "skill_demo" / "examples"
    failed = 0
    for case in CASES:
        result = answer_question(case["question"], skills_root, module=case.get("module"))
        ok = result["status"] == case["expect"]
        failed += 0 if ok else 1
        print("=" * 80)
        print(f"Q: {case['question']}")
        print(f"Expected: {case['expect']} | Actual: {result['status']} | {'OK' if ok else 'FAIL'}")
        print(result["answer"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
