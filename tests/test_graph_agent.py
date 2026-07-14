from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teacher_rag.graph_agent import ask_graph_agent, public_result


SKILLS_ROOT = ROOT / "skill_demo" / "examples"
MISSING_CONFIG = ROOT / "config" / "missing.test.local.json"


class FakeLLM:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return (
            "1. 在新建任务弹窗的执行人选择区找到“执行人”下拉框。\n"
            "2. 选择巡检员；多人协同时点击“+”继续添加。\n"
            "3. 完成后点击“保存并发布”。"
        )

class FailingLLM:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("model unavailable")


class GraphAgentTest(unittest.TestCase):
    def test_skill_match_path_uses_llm_summary(self) -> None:
        state = ask_graph_agent(
            "创建巡检任务时，如何指派执行人？",
            module="任务管理",
            skills_root=SKILLS_ROOT,
            llm_config=MISSING_CONFIG,
            use_llm=True,
            llm_client=FakeLLM(),
        )
        result = public_result(state)

        self.assertEqual(result["status"], "skill_answer")
        self.assertEqual(result["answer_mode"], "llm")
        self.assertIn("执行人", result["answer"])
        self.assertEqual(result["next_action"], None)
        self.assertEqual(result["matches"][0]["name"], "create-inspection-task")
        self.assertEqual(result["materials"][0]["source_type"], "skill")
        self.assertEqual(
            result["trace"],
            [
                "load_context",
                "retrieve_skill",
                "build_skill_material",
                "llm_summarize:llm",
                "final",
            ],
        )

    def test_skill_no_match_returns_rag_required(self) -> None:
        state = ask_graph_agent(
            "怎么修改个人头像？",
            module="个人中心",
            skills_root=SKILLS_ROOT,
            llm_config=MISSING_CONFIG,
            use_llm=True,
            llm_client=FakeLLM(),
        )
        result = public_result(state)

        self.assertEqual(result["status"], "rag_required")
        self.assertEqual(result["answer_mode"], "rag_required")
        self.assertEqual(result["next_action"], "ragflow_retrieve")
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["materials"], [])
        self.assertEqual(result["trace"], ["load_context", "retrieve_skill", "rag_required", "final"])

    def test_without_llm_uses_template_answer(self) -> None:
        state = ask_graph_agent(
            "怎么建巡检任务？",
            module="任务管理",
            skills_root=SKILLS_ROOT,
            llm_config=MISSING_CONFIG,
            use_llm=False,
        )
        result = public_result(state)

        self.assertEqual(result["status"], "skill_answer")
        self.assertEqual(result["answer_mode"], "template")
        self.assertIn("操作步骤", result["answer"])

    def test_llm_failure_falls_back_to_template(self) -> None:
        state = ask_graph_agent(
            "怎么建巡检任务？",
            module="任务管理",
            skills_root=SKILLS_ROOT,
            llm_config=MISSING_CONFIG,
            use_llm=True,
            llm_client=FailingLLM(),
        )
        result = public_result(state)

        self.assertEqual(result["status"], "skill_answer")
        self.assertEqual(result["answer_mode"], "template_fallback")
        self.assertIn("model unavailable", result["llm_error"])
        self.assertIn("操作步骤", result["answer"])


if __name__ == "__main__":
    unittest.main()
