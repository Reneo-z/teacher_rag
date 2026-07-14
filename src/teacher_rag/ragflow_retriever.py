"""Standalone RAGFlow retriever for testing dataset chunk retrieval."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from teacher_rag.skill_agent import RetrievedMaterial, normalize_optional_str


DEFAULT_RAGFLOW_CONFIG_PATH = Path("config/ragflow.local.json")


@dataclass(frozen=True)
class RagflowRetrievalConfig:
    enabled: bool
    base_url: str
    api_key: str | None
    default_dataset_ids: list[str]
    module_dataset_map: dict[str, list[str]]
    page: int = 1
    page_size: int = 5
    similarity_threshold: float = 0.2
    vector_similarity_weight: float = 0.3
    top_k: int = 1024
    keyword: bool = True
    max_chunks_for_llm: int = 4
    min_similarity_for_llm: float = 0.3
    max_chunk_chars: int = 1200
    timeout_seconds: int = 60

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RagflowRetrievalConfig":
        retrieval = config.get("retrieval") or {}
        if not isinstance(retrieval, dict):
            raise ValueError("ragflow retrieval config must be an object")
        module_dataset_map = config.get("module_dataset_map") or {}
        if not isinstance(module_dataset_map, dict):
            raise ValueError("module_dataset_map must be an object")
        return cls(
            enabled=bool(config.get("enabled", False)),
            base_url=str(config.get("base_url") or "http://127.0.0.1:9380"),
            api_key=normalize_optional_str(config.get("api_key")),
            default_dataset_ids=[str(item) for item in config.get("default_dataset_ids") or []],
            module_dataset_map={
                str(key): [str(item) for item in value]
                for key, value in module_dataset_map.items()
            },
            page=int(retrieval.get("page") or 1),
            page_size=int(retrieval.get("page_size") or 5),
            similarity_threshold=float(retrieval.get("similarity_threshold") or 0.2),
            vector_similarity_weight=float(retrieval.get("vector_similarity_weight") or 0.3),
            top_k=int(retrieval.get("top_k") or 1024),
            keyword=bool(retrieval.get("keyword", True)),
            max_chunks_for_llm=int(retrieval.get("max_chunks_for_llm") or 4),
            min_similarity_for_llm=float(retrieval.get("min_similarity_for_llm") or 0.3),
            max_chunk_chars=int(retrieval.get("max_chunk_chars") or 1200),
            timeout_seconds=int(config.get("timeout_seconds") or retrieval.get("timeout_seconds") or 60),
        )

    def dataset_ids_for_module(self, module: str | None) -> list[str]:
        if module and module in self.module_dataset_map:
            return self.module_dataset_map[module]
        return self.default_dataset_ids


@dataclass(frozen=True)
class RagflowChunk:
    content: str
    chunk_id: str | None
    document_id: str | None
    document_name: str | None
    dataset_id: str | None
    similarity: float | None
    raw: dict[str, Any]

    @property
    def title(self) -> str:
        return self.document_name or self.document_id or self.chunk_id or "RAGFlow chunk"


def load_ragflow_config(path: str | Path = DEFAULT_RAGFLOW_CONFIG_PATH) -> RagflowRetrievalConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"RAGFlow config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"RAGFlow config must be a JSON object: {config_path}")
    return RagflowRetrievalConfig.from_dict(config)


def extract_chunks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data", payload)
    if isinstance(data, dict):
        chunks = data.get("chunks") or data.get("documents") or data.get("items") or []
    elif isinstance(data, list):
        chunks = data
    else:
        chunks = []
    if not isinstance(chunks, list):
        raise ValueError(f"Unexpected RAGFlow chunks shape: {type(chunks).__name__}")
    return [item for item in chunks if isinstance(item, dict)]


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def parse_chunk(raw: dict[str, Any]) -> RagflowChunk:
    content = first_present(raw, ["content", "text", "chunk", "page_content", "content_with_weight"])
    doc_name = first_present(raw, ["document_name", "doc_name", "name", "document_title"])
    similarity = first_present(raw, ["similarity", "score", "similarity_score", "rank_score"])
    return RagflowChunk(
        content=str(content or ""),
        chunk_id=normalize_optional_str(first_present(raw, ["chunk_id", "id"])),
        document_id=normalize_optional_str(first_present(raw, ["document_id", "doc_id"])),
        document_name=normalize_optional_str(doc_name),
        dataset_id=normalize_optional_str(first_present(raw, ["dataset_id", "kb_id"])),
        similarity=float(similarity) if similarity is not None else None,
        raw=raw,
    )


class RagflowRetriever:
    def __init__(self, config: RagflowRetrievalConfig) -> None:
        self.config = config

    @classmethod
    def from_config_file(cls, path: str | Path = DEFAULT_RAGFLOW_CONFIG_PATH) -> "RagflowRetriever":
        return cls(load_ragflow_config(path))

    def retrieve(
        self,
        question: str,
        *,
        module: str | None = None,
        dataset_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        page_size: int | None = None,
    ) -> list[RagflowChunk]:
        selected_dataset_ids = dataset_ids or self.config.dataset_ids_for_module(module)
        if not selected_dataset_ids:
            raise ValueError("No RAGFlow dataset_ids configured for this query")
        payload: dict[str, Any] = {
            "question": question,
            "dataset_ids": selected_dataset_ids,
            "page": self.config.page,
            "page_size": page_size or self.config.page_size,
            "similarity_threshold": self.config.similarity_threshold,
            "vector_similarity_weight": self.config.vector_similarity_weight,
            "top_k": self.config.top_k,
            "keyword": self.config.keyword,
        }
        if document_ids:
            payload["document_ids"] = document_ids

        response = self._post_json("/api/v1/retrieval", payload)
        return [parse_chunk(item) for item in extract_chunks(response)]

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = self.config.base_url.rstrip("/") + path
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RAGFlow HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"RAGFlow request failed: {exc.reason}") from exc

        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("code") not in (None, 0, "0"):
            raise RuntimeError(f"RAGFlow API error: {parsed}")
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Unexpected RAGFlow response shape: {raw[:500]}")
        return parsed



def trim_chunk_content(chunk: RagflowChunk, max_chars: int) -> RagflowChunk:
    if max_chars <= 0 or len(chunk.content) <= max_chars:
        return chunk
    return RagflowChunk(
        content=chunk.content[:max_chars].rstrip() + "...",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        dataset_id=chunk.dataset_id,
        similarity=chunk.similarity,
        raw=chunk.raw,
    )


def select_chunks_for_llm(
    chunks: list[RagflowChunk],
    config: RagflowRetrievalConfig,
) -> list[RagflowChunk]:
    candidates = [chunk for chunk in chunks if chunk.content.strip()]
    filtered = [
        chunk
        for chunk in candidates
        if chunk.similarity is None or chunk.similarity >= config.min_similarity_for_llm
    ]
    ranked = sorted(
        filtered,
        key=lambda chunk: chunk.similarity if chunk.similarity is not None else -1.0,
        reverse=True,
    )
    selected = ranked[: config.max_chunks_for_llm]
    return [trim_chunk_content(chunk, config.max_chunk_chars) for chunk in selected]

def chunks_to_materials(chunks: list[RagflowChunk]) -> list[RetrievedMaterial]:
    return [
        RetrievedMaterial(
            source_type="ragflow",
            title=chunk.title,
            content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "dataset_id": chunk.dataset_id,
                "similarity": chunk.similarity,
                "source": "ragflow",
            },
        )
        for chunk in chunks
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test standalone RAGFlow chunk retrieval.")
    parser.add_argument("question", help="Question used for RAGFlow retrieval.")
    parser.add_argument("--config", default=str(DEFAULT_RAGFLOW_CONFIG_PATH))
    parser.add_argument("--module", default=None)
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids")
    parser.add_argument("--document-id", action="append", dest="document_ids")
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--for-llm", action="store_true", help="Filter chunks using max_chunks_for_llm/min_similarity_for_llm/max_chunk_chars before output.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    return parser


def chunk_to_public_dict(chunk: RagflowChunk, *, show_raw: bool = False) -> dict[str, Any]:
    output: dict[str, Any] = {
        "title": chunk.title,
        "content": chunk.content,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "dataset_id": chunk.dataset_id,
        "similarity": chunk.similarity,
    }
    if show_raw:
        output["raw"] = chunk.raw
    return output


def print_human(chunks: list[RagflowChunk]) -> None:
    if not chunks:
        print("未检索到相关 chunks。")
        return
    for idx, chunk in enumerate(chunks, start=1):
        print("=" * 80)
        print(f"Chunk {idx}: {chunk.title}")
        print(f"chunk_id={chunk.chunk_id} document_id={chunk.document_id} dataset_id={chunk.dataset_id} similarity={chunk.similarity}")
        print("-" * 80)
        print(chunk.content.strip() or "<empty content>")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    retriever = RagflowRetriever.from_config_file(args.config)
    chunks = retriever.retrieve(
        args.question,
        module=args.module,
        dataset_ids=args.dataset_ids,
        document_ids=args.document_ids,
        page_size=args.page_size,
    )
    raw_count = len(chunks)
    if args.for_llm:
        chunks = select_chunks_for_llm(chunks, retriever.config)
    if args.json:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "module": args.module,
                    "count": len(chunks),
                    "raw_count": raw_count,
                    "for_llm": args.for_llm,
                    "chunks": [chunk_to_public_dict(chunk, show_raw=args.show_raw) for chunk in chunks],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_human(chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
