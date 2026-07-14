from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teacher_rag.graph_skill_agent import ask_graph_skill_agent, parse_selection_json, public_result
from teacher_rag.ragflow_retriever import RagflowChunk, RagflowRetrievalConfig


SKILLS_ROOT = ROOT / "skill_demo" / "examples"
MISSING_CONFIG = ROOT / "config" / "missing.test.local.json"


class RoutingAndAnswerLLM:
    def __init__(self, *, selected: str | None = "create-inspection-task", confidence: float = 0.91) -> None:
        self.selected = selected
        self.confidence = confidence
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if "上下文抽取节点" in system_prompt:
            module = "任务管理" if "任务管理模块" in user_prompt or "任务管理" in user_prompt else None
            return json.dumps(
                {
                    "module": module,
                    "page": None,
                    "confidence": 0.92 if module else 0.0,
                    "reason": "用户自然语言中明确提到任务管理模块" if module else "用户未明确提到模块",
                },
                ensure_ascii=False,
            )
        if "Skill 路由节点" in system_prompt:
            return json.dumps(
                {
                    "selected_skill_name": self.selected,
                    "confidence": self.confidence,
                    "reason": "用户询问巡检任务中的执行人指派，候选 Skill 覆盖创建巡检任务和执行人选择。",
                },
                ensure_ascii=False,
            )
        return (
            "1. 在新建任务弹窗中找到“执行人”下拉框。\n"
            "2. 选择巡检员；多人协同时点击“+”继续添加。\n"
            "3. 完成后点击“保存并发布”。"
        )


class FakeRagflowRetriever:
    def __init__(self, chunks: list[RagflowChunk] | None = None, *, should_fail: bool = False) -> None:
        self.config = RagflowRetrievalConfig(
            enabled=True,
            base_url="http://ragflow.test",
            api_key=None,
            default_dataset_ids=["dataset-1"],
            module_dataset_map={"个人中心": ["dataset-1"]},
            max_chunks_for_llm=2,
            min_similarity_for_llm=0.3,
            max_chunk_chars=80,
        )
        self.chunks = chunks or []
        self.should_fail = should_fail
        self.calls: list[tuple[str, str | None]] = []

    def retrieve(self, question: str, *, module: str | None = None, **kwargs):
        self.calls.append((question, module))
        if self.should_fail:
            raise RuntimeError("ragflow unavailable")
        return self.chunks


def make_chunk(content: str, similarity: float, chunk_id: str) -> RagflowChunk:
    return RagflowChunk(
        content=content,
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name="faq.md",
        dataset_id="dataset-1",
        similarity=similarity,
        raw={"id": chunk_id, "content": content, "similarity": similarity},
    )


class BrokenSelectionLLM:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return "not json"


class GraphSkillAgentTest(unittest.TestCase):
    def test_llm_selects_skill_then_summarizes_answer(self) -> None:
        llm = RoutingAndAnswerLLM()

        state = ask_graph_skill_agent(
            "创建巡检任务时，如何指派执行人？",
            module="任务管理",
            skills_root=SKILLS_ROOT,
            llm_config=MISSING_CONFIG,
            llm_client=llm,
        )
        result = public_result(state)

        self.assertEqual(result["status"], "skill_answer")
        self.assertEqual(result["answer_mode"], "llm")
        self.assertEqual(result["selected_skill_name"], "create-inspection-task")
        self.assertGreater(result["selection_confidence"], 0.55)
        self.assertIn("执行人", result["answer"])
        self.assertEqual(result["matches"][0]["name"], "create-inspection-task")
        self.assertEqual(
            result["trace"],
            [
                "load_context",
                "load_skill_catalog",
                "extract_context:explicit",
                "select_skill",
                "build_skill_material",
                "llm_summarize:llm",
                "final",
            ],
        )
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("候选 Skill catalog", llm.calls[0][1])


    def test_extracts_module_from_natural_language_when_not_explicit(self) -> None:
        llm = RoutingAndAnswerLLM()

        state = ask_graph_skill_agent(
            "我在任务管理模块中想知道在创建巡检任务时，如何指派执行人？",
            skills_root=SKILLS_ROOT,
            llm_config=MISSING_CONFIG,
            llm_client=llm,
        )
        result = public_result(state)

        self.assertEqual(result["module"], "任务管理")
        self.assertEqual(result["extracted_module"], "任务管理")
        self.assertGreaterEqual(result["context_extraction_confidence"], 0.5)
        self.assertEqual(result["status"], "skill_answer")
        self.assertEqual(
            result["trace"],
            [
                "load_context",
                "load_skill_catalog",
                "extract_context",
                "select_skill",
                "build_skill_material",
                "llm_summarize:llm",
                "final",
            ],
        )

    def test_llm_no_selection_routes_to_ragflow_then_summarizes(self) -> None:
        llm = RoutingAndAnswerLLM(selected=None, confidence=0.2)
        retriever = FakeRagflowRetriever(
            [
                make_chunk("头像修改流程：进入个人中心，点击头像，上传新图片后保存。", 0.8, "c1"),
                make_chunk("低相关内容", 0.1, "c2"),
                make_chunk("个人资料保存后刷新页面验证。", 0.6, "c3"),
            ]
        )

        state = ask_graph_skill_agent(
            "怎么修改个人头像？",
            module="个人中心",
            skills_root=SKILLS_ROOT,
            llm_config=MISSING_CONFIG,
            llm_client=llm,
            ragflow_retriever=retriever,
        )
        result = public_result(state)

        self.assertEqual(result["status"], "rag_answer")
        self.assertEqual(result["answer_mode"], "llm")
        self.assertEqual(result["next_action"], None)
        self.assertEqual(result["selected_skill_name"], None)
        self.assertEqual(result["matches"], [])
        self.assertEqual(len(result["materials"]), 2)
        self.assertEqual(result["materials"][0]["source_type"], "ragflow")
        self.assertEqual(result["raw_rag_chunk_count"], 3)
        self.assertEqual(retriever.calls, [("怎么修改个人头像？", "个人中心")])
        self.assertEqual(
            result["trace"],
            [
                "load_context",
                "load_skill_catalog",
                "extract_context:explicit",
                "select_skill",
                "ragflow_retrieve",
                "build_rag_material",
                "llm_summarize:llm",
                "final",
            ],
        )

    def test_llm_no_selection_and_empty_ragflow_returns_not_found(self) -> None:
        llm = RoutingAndAnswerLLM(selected=None, confidence=0.2)
        retriever = FakeRagflowRetriever([])

        state = ask_graph_skill_agent(
            "怎么申请食堂饭卡？",
            module="个人中心",
            skills_root=SKILLS_ROOT,
            llm_config=MISSING_CONFIG,
            llm_client=llm,
            ragflow_retriever=retriever,
        )
        result = public_result(state)

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["answer_mode"], "not_found")
        self.assertEqual(result["next_action"], "add_skill_or_knowledge")
        self.assertEqual(result["raw_rag_chunk_count"], 0)
        self.assertEqual(result["trace"], ["load_context", "load_skill_catalog", "extract_context:explicit", "select_skill", "ragflow_retrieve:not_found", "final"])

    def test_ragflow_error_is_reported(self) -> None:
        llm = RoutingAndAnswerLLM(selected=None, confidence=0.2)
        retriever = FakeRagflowRetriever(should_fail=True)

        state = ask_graph_skill_agent(
            "怎么修改个人头像？",
            module="个人中心",
            skills_root=SKILLS_ROOT,
            llm_config=MISSING_CONFIG,
            llm_client=llm,
            ragflow_retriever=retriever,
        )
        result = public_result(state)

        self.assertEqual(result["status"], "ragflow_error")
        self.assertEqual(result["answer_mode"], "ragflow_error")
        self.assertEqual(result["next_action"], "inspect_ragflow_config")
        self.assertIn("ragflow unavailable", result["llm_error"])

    def test_selection_parse_accepts_json_inside_text(self) -> None:
        parsed = parse_selection_json('```json\n{"selected_skill_name": null, "confidence": 0, "reason": "无"}\n```')

        self.assertIsNone(parsed["selected_skill_name"])
        self.assertEqual(parsed["confidence"], 0)

    def test_selection_error_stops_at_final(self) -> None:
        state = ask_graph_skill_agent(
            "怎么建巡检任务？",
            module="任务管理",
            skills_root=SKILLS_ROOT,
            llm_config=MISSING_CONFIG,
            llm_client=BrokenSelectionLLM(),
        )
        result = public_result(state)

        self.assertEqual(result["status"], "skill_selection_error")
        self.assertEqual(result["answer_mode"], "skill_selection_error")
        self.assertEqual(result["next_action"], "inspect_llm_config")
        self.assertIn("not JSON", result["llm_error"])
        self.assertEqual(result["trace"], ["load_context", "load_skill_catalog", "extract_context:explicit", "select_skill:error", "final"])


if __name__ == "__main__":
    unittest.main()
