from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teacher_rag.skill_agent import (
    OpenAICompatibleChatLLM,
    SkillRouter,
    answer_question,
    build_env_llm_client,
    load_llm_config,
    load_skills,
)


class RecordingLLM:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return (
            "1. 在新建任务弹窗中找到“执行人”下拉框。\n"
            "2. 选择巡检员；如需多人协同，点击“+”继续添加。\n"
            "3. 确认后继续保存并发布任务。"
        )


SKILLS_ROOT = ROOT / "skill_demo" / "examples"


class SkillAgentTest(unittest.TestCase):
    def test_loads_demo_skill(self) -> None:
        skills = load_skills(SKILLS_ROOT)

        self.assertEqual(len(skills), 1)
        skill = skills[0]
        self.assertEqual(skill.name, "create-inspection-task")
        self.assertEqual(skill.display_name, "创建巡检任务")
        self.assertEqual(skill.module, "任务管理")
        self.assertEqual(len(skill.steps), 6)
        self.assertIn("页面出现“任务创建成功”提示。", skill.verification)

    def test_routes_typical_question_to_skill(self) -> None:
        router = SkillRouter(load_skills(SKILLS_ROOT))

        matches = router.match("我想给设备安排一次巡检，应该在哪里派单？", module="任务管理")

        self.assertTrue(matches)
        self.assertEqual(matches[0].skill.name, "create-inspection-task")
        self.assertGreaterEqual(matches[0].score, 3.0)

    def test_renders_step_answer_for_matched_skill(self) -> None:
        result = answer_question("怎么建巡检任务？", SKILLS_ROOT, module="任务管理")

        self.assertEqual(result["status"], "skill_answer")
        self.assertIn("操作步骤", result["answer"])
        self.assertIn("保存并发布", result["answer"])
        self.assertIn("完成后这样验证", result["answer"])


    def test_uses_llm_to_focus_on_sub_step_question(self) -> None:
        llm = RecordingLLM()

        result = answer_question(
            "创建巡检任务时，如何指派执行人？",
            SKILLS_ROOT,
            module="任务管理",
            llm_client=llm,
        )

        self.assertEqual(result["status"], "skill_answer")
        self.assertEqual(result["answer_mode"], "llm")
        self.assertIn("执行人", result["answer"])
        self.assertNotIn("进入新建任务页面", result["answer"])
        self.assertIn("只输出和该子步骤直接相关的流程", llm.system_prompt)
        self.assertIn("创建巡检任务时，如何指派执行人？", llm.user_prompt)
        self.assertIn("指派执行人", llm.user_prompt)


    def test_loads_llm_config_file(self) -> None:
        config_path = ROOT / "config" / "llm.test.local.json"
        config_path.write_text(
            '{"enabled": true, "base_url": "http://127.0.0.1:8000/v1", '
            '"model": "local-test", "api_key": "test-key", "timeout_seconds": 7}',
            encoding="utf-8",
        )
        try:
            config = load_llm_config(config_path)
            client = build_env_llm_client(config_path=config_path)
        finally:
            config_path.unlink(missing_ok=True)

        self.assertEqual(config["model"], "local-test")
        self.assertIsInstance(client, OpenAICompatibleChatLLM)
        self.assertEqual(client.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(client.model, "local-test")
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.timeout_seconds, 7)

    def test_returns_no_match_for_uncovered_question(self) -> None:
        result = answer_question("如何修改个人头像？", SKILLS_ROOT)

        self.assertEqual(result["status"], "no_skill_match")
        self.assertIn("RAGFlow", result["answer"])


if __name__ == "__main__":
    unittest.main()

