from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teacher_rag.ragflow_retriever import (
    RagflowRetriever,
    RagflowRetrievalConfig,
    chunks_to_materials,
    extract_chunks,
    select_chunks_for_llm,
    parse_chunk,
)


class FakeResponse:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.body


class RagflowRetrieverTest(unittest.TestCase):
    def test_extracts_chunks_from_standard_payload(self) -> None:
        payload = {
            "code": 0,
            "data": {
                "chunks": [
                    {
                        "content": "执行人为空时，请检查人员权限。",
                        "id": "chunk-1",
                        "document_id": "doc-1",
                        "doc_name": "faq.md",
                        "dataset_id": "dataset-1",
                        "similarity": 0.82,
                    }
                ]
            },
        }

        chunks = [parse_chunk(item) for item in extract_chunks(payload)]

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "执行人为空时，请检查人员权限。")
        self.assertEqual(chunks[0].chunk_id, "chunk-1")
        self.assertEqual(chunks[0].document_name, "faq.md")
        self.assertEqual(chunks[0].similarity, 0.82)

    def test_converts_chunks_to_retrieved_materials(self) -> None:
        chunk = parse_chunk(
            {
                "content": "先检查执行人角色，再检查排班。",
                "id": "chunk-2",
                "doc_name": "执行人FAQ.md",
                "dataset_id": "dataset-1",
                "score": 0.75,
            }
        )

        materials = chunks_to_materials([chunk])

        self.assertEqual(materials[0].source_type, "ragflow")
        self.assertEqual(materials[0].title, "执行人FAQ.md")
        self.assertIn("检查执行人角色", materials[0].content)
        self.assertEqual(materials[0].metadata["source"], "ragflow")


    def test_select_chunks_for_llm_filters_sorts_limits_and_trims(self) -> None:
        config = RagflowRetrievalConfig(
            enabled=True,
            base_url="http://ragflow.local",
            api_key=None,
            default_dataset_ids=["dataset-1"],
            module_dataset_map={},
            max_chunks_for_llm=2,
            min_similarity_for_llm=0.3,
            max_chunk_chars=6,
        )
        chunks = [
            parse_chunk({"content": "低分内容", "id": "low", "similarity": 0.1}),
            parse_chunk({"content": "第二高相关内容很长", "id": "mid", "similarity": 0.6}),
            parse_chunk({"content": "最高相关内容也很长", "id": "high", "similarity": 0.9}),
        ]

        selected = select_chunks_for_llm(chunks, config)

        self.assertEqual([chunk.chunk_id for chunk in selected], ["high", "mid"])
        self.assertTrue(selected[0].content.endswith("..."))
        self.assertLessEqual(len(selected[0].content), 9)

    def test_retrieve_posts_expected_payload(self) -> None:
        config = RagflowRetrievalConfig(
            enabled=True,
            base_url="http://ragflow.local",
            api_key="test-key",
            default_dataset_ids=["default-ds"],
            module_dataset_map={"任务管理": ["task-ds"]},
            page_size=3,
        )
        retriever = RagflowRetriever(config)
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = request.data.decode("utf-8")
            captured["timeout"] = timeout
            return FakeResponse(
                '{"code": 0, "data": {"chunks": [{"content": "chunk text", "id": "c1"}]}}'
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            chunks = retriever.retrieve("执行人为空怎么办？", module="任务管理")

        self.assertEqual(captured["url"], "http://ragflow.local/api/v1/retrieval")
        self.assertIn("Bearer test-key", captured["headers"].get("Authorization", ""))
        self.assertIn('"dataset_ids": ["task-ds"]', captured["body"])
        self.assertEqual(chunks[0].content, "chunk text")


if __name__ == "__main__":
    unittest.main()
