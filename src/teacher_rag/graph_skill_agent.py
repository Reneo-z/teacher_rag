"""LangGraph-native Skill/RAGFlow routing tutorial agent.

This graph keeps the earlier keyword router available for comparison. It uses
LLM nodes to extract context, select Skills from catalog metadata, and summarize
retrieved materials from either Skill or RAGFlow.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - exercised when dependency is absent.
    raise ImportError(
        "langgraph is required for graph_skill_agent. Install it with `.venv/bin/python -m pip install -r requirements.txt`."
    ) from exc

from teacher_rag.ragflow_retriever import (
    DEFAULT_RAGFLOW_CONFIG_PATH,
    RagflowChunk,
    RagflowRetriever,
    chunks_to_materials,
    select_chunks_for_llm,
)
from teacher_rag.skill_agent import (
    DEFAULT_LLM_CONFIG_PATH,
    LLMClient,
    RetrievedMaterial,
    Skill,
    SkillMatch,
    build_env_llm_client,
    build_skill_material,
    generate_llm_tutorial_answer,
    load_skills,
    render_tutorial_answer,
)


SELECTION_THRESHOLD = 0.55
CONTEXT_EXTRACTION_THRESHOLD = 0.5


class GraphSkillAgentState(TypedDict):
    question: str
    module: NotRequired[str | None]
    page: NotRequired[str | None]
    explicit_module: NotRequired[str | None]
    explicit_page: NotRequired[str | None]
    extracted_module: NotRequired[str | None]
    extracted_page: NotRequired[str | None]
    context_extraction_confidence: NotRequired[float]
    context_extraction_reason: NotRequired[str]
    skills_root: NotRequired[str]
    llm_config: NotRequired[str]
    ragflow_config: NotRequired[str]
    use_llm: NotRequired[bool]
    status: NotRequired[str]
    answer: NotRequired[str]
    answer_mode: NotRequired[str]
    llm_error: NotRequired[str | None]
    next_action: NotRequired[str | None]
    trace: NotRequired[list[str]]
    skills: NotRequired[list[Skill]]
    skill_catalog: NotRequired[list[dict[str, Any]]]
    selected_skill_name: NotRequired[str | None]
    selection_confidence: NotRequired[float]
    selection_reason: NotRequired[str]
    selection_raw: NotRequired[str]
    skill_matches: NotRequired[list[SkillMatch]]
    matches: NotRequired[list[dict[str, Any]]]
    materials: NotRequired[list[RetrievedMaterial]]
    material_summaries: NotRequired[list[dict[str, Any]]]
    rag_chunks: NotRequired[list[RagflowChunk]]
    raw_rag_chunk_count: NotRequired[int]


def append_trace(state: GraphSkillAgentState, event: str) -> list[str]:
    return [*state.get("trace", []), event]


def serialize_matches(matches: list[SkillMatch]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.skill.name,
            "display_name": item.skill.display_name,
            "score": item.score,
            "reasons": item.reasons,
            "path": str(item.skill.path),
        }
        for item in matches
    ]


def serialize_materials(materials: list[RetrievedMaterial]) -> list[dict[str, Any]]:
    return [
        {
            "source_type": item.source_type,
            "title": item.title,
            "metadata": item.metadata,
        }
        for item in materials
    ]


def build_skill_catalog(skills: list[Skill]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for skill in skills:
        catalog.append(
            {
                "name": skill.name,
                "display_name": skill.display_name,
                "system": skill.system,
                "module": skill.module,
                "description": skill.description,
                "keywords": skill.keywords,
                "prerequisites_summary": skill.metadata.get("prerequisites_summary", ""),
                "path": str(skill.path),
            }
        )
    return catalog


def get_llm_client(
    state: GraphSkillAgentState,
    explicit_client: LLMClient | None,
    *,
    force: bool = True,
) -> LLMClient | None:
    return explicit_client or build_env_llm_client(
        force=force or bool(state.get("use_llm", False)),
        config_path=state.get("llm_config") or DEFAULT_LLM_CONFIG_PATH,
    )


def load_context_node(state: GraphSkillAgentState) -> dict[str, Any]:
    return {
        "skills_root": state.get("skills_root") or "skill_demo/examples",
        "llm_config": state.get("llm_config") or str(DEFAULT_LLM_CONFIG_PATH),
        "ragflow_config": state.get("ragflow_config") or str(DEFAULT_RAGFLOW_CONFIG_PATH),
        "use_llm": bool(state.get("use_llm", False)),
        "explicit_module": state.get("module"),
        "explicit_page": state.get("page"),
        "status": "context_loaded",
        "trace": append_trace(state, "load_context"),
    }


def load_skill_catalog_node(state: GraphSkillAgentState) -> dict[str, Any]:
    skills = load_skills(state.get("skills_root") or "skill_demo/examples")
    catalog = build_skill_catalog(skills)
    return {
        "skills": skills,
        "skill_catalog": catalog,
        "status": "skill_catalog_loaded",
        "trace": append_trace(state, "load_skill_catalog"),
    }


def parse_llm_json(raw: str, *, label: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"{label} response is not JSON: {raw[:300]}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} response must be a JSON object")
    return parsed


def parse_selection_json(raw: str) -> dict[str, Any]:
    return parse_llm_json(raw, label="Skill selection")


def candidate_modules_from_state(state: GraphSkillAgentState) -> list[str]:
    modules: list[str] = []
    for item in state.get("skill_catalog") or []:
        module = item.get("module")
        if module and module not in modules:
            modules.append(str(module))
    try:
        retriever = RagflowRetriever.from_config_file(
            state.get("ragflow_config") or DEFAULT_RAGFLOW_CONFIG_PATH
        )
        for module in retriever.config.module_dataset_map:
            if module not in modules:
                modules.append(module)
    except Exception:
        pass
    return modules


def extract_context_node_factory(llm_client: LLMClient | None = None):
    def extract_context_node(state: GraphSkillAgentState) -> dict[str, Any]:
        if state.get("module") or state.get("page"):
            return {
                "explicit_module": state.get("module"),
                "explicit_page": state.get("page"),
                "status": "context_extracted",
                "trace": append_trace(state, "extract_context:explicit"),
            }

        client = get_llm_client(state, llm_client, force=True)
        if not client:
            return {
                "status": "context_extraction_unavailable",
                "trace": append_trace(state, "extract_context:no_llm"),
            }

        candidate_modules = candidate_modules_from_state(state)
        response_schema = {
            "module": "string|null，只能从候选模块中选择",
            "page": "string|null",
            "confidence": "0到1的小数",
            "reason": "简短理由",
        }
        system_prompt = (
            "你是教程智能体的上下文抽取节点。请从用户自然语言问题中抽取用户明确提到的业务模块和页面。"
            "模块必须来自候选模块列表；如果用户没有明确说所在模块，module 返回 null。"
            "必须返回严格 JSON，不要输出其他文字。"
        )
        user_prompt = (
            f"用户问题：{state['question']}\n"
            f"候选模块：{json.dumps(candidate_modules, ensure_ascii=False)}\n"
            "返回 JSON 格式："
            f"{json.dumps(response_schema, ensure_ascii=False)}"
        )
        try:
            raw = client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = parse_llm_json(raw, label="Context extraction")
            module = parsed.get("module")
            page = parsed.get("page")
            confidence = float(parsed.get("confidence") or 0.0)
            if module is not None:
                module = str(module).strip() or None
            if page is not None:
                page = str(page).strip() or None
            if module and module not in candidate_modules:
                module = None
                confidence = 0.0
            if confidence < CONTEXT_EXTRACTION_THRESHOLD:
                module = None
                page = None
            return {
                "module": module,
                "page": page,
                "extracted_module": module,
                "extracted_page": page,
                "context_extraction_confidence": confidence,
                "context_extraction_reason": str(parsed.get("reason") or ""),
                "status": "context_extracted",
                "trace": append_trace(state, "extract_context"),
            }
        except Exception as exc:  # noqa: BLE001 - extraction failure should not stop routing.
            return {
                "module": None,
                "page": None,
                "extracted_module": None,
                "extracted_page": None,
                "context_extraction_confidence": 0.0,
                "context_extraction_reason": f"context extraction failed: {exc}",
                "status": "context_extraction_error",
                "trace": append_trace(state, "extract_context:error"),
            }

    return extract_context_node


def select_skill_node_factory(llm_client: LLMClient | None = None):
    def select_skill_node(state: GraphSkillAgentState) -> dict[str, Any]:
        client = get_llm_client(state, llm_client, force=True)
        if not client:
            return {
                "status": "skill_selection_unavailable",
                "answer": "未启用大模型，无法使用 LangGraph 原生 Skill 选择流程。",
                "answer_mode": "skill_selection_unavailable",
                "next_action": "enable_llm",
                "trace": append_trace(state, "select_skill:no_llm"),
            }

        catalog = state.get("skill_catalog") or []
        system_prompt = (
            "你是教程智能体的 Skill 路由节点。你需要根据用户问题、当前模块/页面上下文和候选 Skill catalog，"
            "选择最适合回答问题的一个 Skill。只允许从候选 Skill 的 name 中选择；如果没有足够匹配的 Skill，"
            "selected_skill_name 返回 null。必须返回严格 JSON，不要输出其他文字。"
        )
        response_schema = {
            "selected_skill_name": "string|null",
            "confidence": "0到1的小数",
            "reason": "简短理由",
        }
        user_prompt = (
            f"用户问题：{state['question']}\n"
            f"当前模块：{state.get('module') or ''}\n"
            f"当前页面：{state.get('page') or ''}\n\n"
            "候选 Skill catalog：\n"
            f"{json.dumps(catalog, ensure_ascii=False, indent=2)}\n\n"
            "返回 JSON 格式："
            f"{json.dumps(response_schema, ensure_ascii=False)}"
        )
        try:
            raw = client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = parse_selection_json(raw)
            selected = parsed.get("selected_skill_name")
            if selected is not None:
                selected = str(selected).strip() or None
            confidence = float(parsed.get("confidence") or 0.0)
            reason = str(parsed.get("reason") or "")
            return {
                "selected_skill_name": selected,
                "selection_confidence": confidence,
                "selection_reason": reason,
                "selection_raw": raw,
                "status": "skill_selected" if selected and confidence >= SELECTION_THRESHOLD else "skill_not_found",
                "trace": append_trace(state, "select_skill"),
            }
        except Exception as exc:  # noqa: BLE001 - keep graph response inspectable.
            return {
                "selected_skill_name": None,
                "selection_confidence": 0.0,
                "selection_reason": "",
                "selection_raw": "",
                "status": "skill_selection_error",
                "answer": "Skill 选择节点调用大模型失败，无法继续匹配。",
                "answer_mode": "skill_selection_error",
                "llm_error": str(exc),
                "next_action": "inspect_llm_config",
                "trace": append_trace(state, "select_skill:error"),
            }

    return select_skill_node


def route_after_selection(state: GraphSkillAgentState) -> Literal["build_skill_material", "ragflow_retrieve", "final"]:
    if state.get("status") in {"skill_selection_error", "skill_selection_unavailable"}:
        return "final"
    selected = state.get("selected_skill_name")
    confidence = float(state.get("selection_confidence") or 0.0)
    if selected and confidence >= SELECTION_THRESHOLD:
        return "build_skill_material"
    return "ragflow_retrieve"


def build_selected_skill_material_node(state: GraphSkillAgentState) -> dict[str, Any]:
    selected = state.get("selected_skill_name")
    skills = state.get("skills") or []
    skill = next((item for item in skills if item.name == selected), None)
    if not skill:
        return {
            "status": "rag_required",
            "answer": "大模型选择的 Skill 不在本地 catalog 中，需要进入 RAGFlow 兜底。",
            "answer_mode": "rag_required",
            "next_action": "ragflow_retrieve",
            "trace": append_trace(state, "build_skill_material:selected_missing"),
        }

    match = SkillMatch(
        skill=skill,
        score=round(float(state.get("selection_confidence") or 0.0) * 10, 3),
        reasons=[state.get("selection_reason") or "LLM selected this Skill from catalog"],
    )
    materials = [build_skill_material(match)]
    matches = [match]
    return {
        "skill_matches": matches,
        "matches": serialize_matches(matches),
        "materials": materials,
        "material_summaries": serialize_materials(materials),
        "status": "skill_material_ready",
        "trace": append_trace(state, "build_skill_material"),
    }


def ragflow_retrieve_node_factory(ragflow_retriever: RagflowRetriever | None = None):
    def ragflow_retrieve_node(state: GraphSkillAgentState) -> dict[str, Any]:
        try:
            retriever = ragflow_retriever or RagflowRetriever.from_config_file(
                state.get("ragflow_config") or DEFAULT_RAGFLOW_CONFIG_PATH
            )
            chunks = retriever.retrieve(state["question"], module=state.get("module"))
            selected_chunks = select_chunks_for_llm(chunks, retriever.config)
            if not selected_chunks:
                return {
                    "status": "not_found",
                    "answer": "Skill 未命中，RAGFlow 也没有检索到足够可靠的知识片段。",
                    "answer_mode": "not_found",
                    "next_action": "add_skill_or_knowledge",
                    "rag_chunks": [],
                    "raw_rag_chunk_count": len(chunks),
                    "trace": append_trace(state, "ragflow_retrieve:not_found"),
                }
            return {
                "status": "rag_retrieved",
                "rag_chunks": selected_chunks,
                "raw_rag_chunk_count": len(chunks),
                "trace": append_trace(state, "ragflow_retrieve"),
            }
        except Exception as exc:  # noqa: BLE001 - keep graph response inspectable.
            return {
                "status": "ragflow_error",
                "answer": "Skill 未命中，且 RAGFlow 检索失败。",
                "answer_mode": "ragflow_error",
                "next_action": "inspect_ragflow_config",
                "llm_error": str(exc),
                "trace": append_trace(state, "ragflow_retrieve:error"),
            }

    return ragflow_retrieve_node


def route_after_ragflow(state: GraphSkillAgentState) -> Literal["build_rag_material", "final"]:
    return "build_rag_material" if state.get("status") == "rag_retrieved" else "final"


def build_rag_material_node(state: GraphSkillAgentState) -> dict[str, Any]:
    chunks = state.get("rag_chunks") or []
    materials = chunks_to_materials(chunks)
    return {
        "materials": materials,
        "material_summaries": serialize_materials(materials),
        "status": "rag_material_ready",
        "trace": append_trace(state, "build_rag_material"),
    }


def llm_summarize_node_factory(llm_client: LLMClient | None = None):
    def llm_summarize_node(state: GraphSkillAgentState) -> dict[str, Any]:
        materials = state.get("materials") or []
        matches = state.get("skill_matches") or []
        is_rag_answer = any(item.source_type == "ragflow" for item in materials)
        if not materials:
            return {
                "status": "not_found",
                "answer": "没有可用于总结的资料。",
                "answer_mode": "not_found",
                "next_action": "add_skill_or_knowledge",
                "trace": append_trace(state, "llm_summarize:no_materials"),
            }

        client = get_llm_client(state, llm_client, force=True)
        if client:
            try:
                return {
                    "status": "rag_answer" if is_rag_answer else "skill_answer",
                    "answer": generate_llm_tutorial_answer(
                        question=state["question"],
                        materials=materials,
                        llm_client=client,
                    ),
                    "answer_mode": "llm",
                    "llm_error": None,
                    "next_action": None,
                    "trace": append_trace(state, "llm_summarize:llm"),
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "rag_answer" if is_rag_answer else "skill_answer",
                    "answer": render_tutorial_answer(matches[0], state["question"]) if matches else materials[0].content,
                    "answer_mode": "template_fallback",
                    "llm_error": str(exc),
                    "next_action": None,
                    "trace": append_trace(state, "llm_summarize:fallback"),
                }

        return {
            "status": "rag_answer" if is_rag_answer else "skill_answer",
            "answer": render_tutorial_answer(matches[0], state["question"]) if matches else materials[0].content,
            "answer_mode": "template",
            "llm_error": None,
            "next_action": None,
            "trace": append_trace(state, "llm_summarize:template"),
        }

    return llm_summarize_node


def final_node(state: GraphSkillAgentState) -> dict[str, Any]:
    return {
        "trace": append_trace(state, "final"),
        "matches": state.get("matches", []),
        "material_summaries": state.get("material_summaries", []),
    }


def build_graph_skill_agent(
    llm_client: LLMClient | None = None,
    ragflow_retriever: RagflowRetriever | None = None,
):
    builder = StateGraph(GraphSkillAgentState)
    builder.add_node("load_context", load_context_node)
    builder.add_node("load_skill_catalog", load_skill_catalog_node)
    builder.add_node("extract_context", extract_context_node_factory(llm_client=llm_client))
    builder.add_node("select_skill", select_skill_node_factory(llm_client=llm_client))
    builder.add_node("build_skill_material", build_selected_skill_material_node)
    builder.add_node("ragflow_retrieve", ragflow_retrieve_node_factory(ragflow_retriever=ragflow_retriever))
    builder.add_node("build_rag_material", build_rag_material_node)
    builder.add_node("llm_summarize", llm_summarize_node_factory(llm_client=llm_client))
    builder.add_node("final", final_node)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "load_skill_catalog")
    builder.add_edge("load_skill_catalog", "extract_context")
    builder.add_edge("extract_context", "select_skill")
    builder.add_conditional_edges(
        "select_skill",
        route_after_selection,
        {
            "build_skill_material": "build_skill_material",
            "ragflow_retrieve": "ragflow_retrieve",
            "final": "final",
        },
    )
    builder.add_edge("build_skill_material", "llm_summarize")
    builder.add_edge("llm_summarize", "final")
    builder.add_conditional_edges(
        "ragflow_retrieve",
        route_after_ragflow,
        {
            "build_rag_material": "build_rag_material",
            "final": "final",
        },
    )
    builder.add_edge("build_rag_material", "llm_summarize")
    builder.add_edge("final", END)
    return builder.compile()


def ask_graph_skill_agent(
    question: str,
    *,
    module: str | None = None,
    page: str | None = None,
    skills_root: str | Path = "skill_demo/examples",
    llm_config: str | Path = DEFAULT_LLM_CONFIG_PATH,
    use_llm: bool = True,
    llm_client: LLMClient | None = None,
    ragflow_config: str | Path = DEFAULT_RAGFLOW_CONFIG_PATH,
    ragflow_retriever: RagflowRetriever | None = None,
) -> GraphSkillAgentState:
    graph = build_graph_skill_agent(llm_client=llm_client, ragflow_retriever=ragflow_retriever)
    initial_state: GraphSkillAgentState = {
        "question": question,
        "module": module,
        "page": page,
        "explicit_module": module,
        "explicit_page": page,
        "skills_root": str(skills_root),
        "llm_config": str(llm_config),
        "use_llm": use_llm,
        "ragflow_config": str(ragflow_config),
    }
    return graph.invoke(initial_state)


def public_result(state: GraphSkillAgentState) -> dict[str, Any]:
    return {
        "status": state.get("status"),
        "answer": state.get("answer"),
        "answer_mode": state.get("answer_mode"),
        "llm_error": state.get("llm_error"),
        "next_action": state.get("next_action"),
        "module": state.get("module"),
        "page": state.get("page"),
        "extracted_module": state.get("extracted_module"),
        "extracted_page": state.get("extracted_page"),
        "context_extraction_confidence": state.get("context_extraction_confidence"),
        "context_extraction_reason": state.get("context_extraction_reason"),
        "selected_skill_name": state.get("selected_skill_name"),
        "selection_confidence": state.get("selection_confidence"),
        "selection_reason": state.get("selection_reason"),
        "matches": state.get("matches", []),
        "materials": state.get("material_summaries", []),
        "trace": state.get("trace", []),
        "raw_rag_chunk_count": state.get("raw_rag_chunk_count"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LangGraph-native Skill routing tutorial agent.")
    parser.add_argument("question", help="User question in natural language.")
    parser.add_argument("--skills-root", default="skill_demo/examples")
    parser.add_argument("--module", default=None)
    parser.add_argument("--page", default=None)
    parser.add_argument("--llm-config", default=str(DEFAULT_LLM_CONFIG_PATH))
    parser.add_argument("--ragflow-config", default=str(DEFAULT_RAGFLOW_CONFIG_PATH))
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM config loading; useful only for failure-path checks.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    state = ask_graph_skill_agent(
        args.question,
        module=args.module,
        page=args.page,
        skills_root=args.skills_root,
        llm_config=args.llm_config,
        use_llm=not args.no_llm,
        ragflow_config=args.ragflow_config,
    )
    result = public_result(state)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("answer") or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
