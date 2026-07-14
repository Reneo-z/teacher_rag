"""LangGraph orchestration skeleton for the tutorial agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - exercised when dependency is absent.
    raise ImportError(
        "langgraph is required for graph_agent. Install it with `python3 -m pip install langgraph`."
    ) from exc

from teacher_rag.skill_agent import (
    DEFAULT_LLM_CONFIG_PATH,
    LLMClient,
    RetrievedMaterial,
    SkillMatch,
    SkillRouter,
    build_env_llm_client,
    build_skill_material,
    generate_llm_tutorial_answer,
    load_skills,
    render_tutorial_answer,
)


class TutorialAgentState(TypedDict):
    question: str
    module: NotRequired[str | None]
    page: NotRequired[str | None]
    skills_root: NotRequired[str]
    llm_config: NotRequired[str]
    use_llm: NotRequired[bool]
    status: NotRequired[str]
    answer: NotRequired[str]
    answer_mode: NotRequired[str]
    llm_error: NotRequired[str | None]
    next_action: NotRequired[str | None]
    error: NotRequired[str | None]
    trace: NotRequired[list[str]]
    skill_matches: NotRequired[list[SkillMatch]]
    materials: NotRequired[list[RetrievedMaterial]]
    matches: NotRequired[list[dict[str, Any]]]
    material_summaries: NotRequired[list[dict[str, Any]]]


def append_trace(state: TutorialAgentState, event: str) -> list[str]:
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


def load_context_node(state: TutorialAgentState) -> dict[str, Any]:
    return {
        "skills_root": state.get("skills_root") or "skill_demo/examples",
        "llm_config": state.get("llm_config") or str(DEFAULT_LLM_CONFIG_PATH),
        "use_llm": bool(state.get("use_llm", False)),
        "status": "context_loaded",
        "trace": append_trace(state, "load_context"),
    }


def retrieve_skill_node(state: TutorialAgentState) -> dict[str, Any]:
    skills = load_skills(state.get("skills_root") or "skill_demo/examples")
    matches = SkillRouter(skills).match(
        state["question"],
        module=state.get("module"),
        page=state.get("page"),
    )
    return {
        "skill_matches": matches,
        "matches": serialize_matches(matches),
        "status": "skill_matched" if matches else "skill_not_found",
        "trace": append_trace(state, "retrieve_skill"),
    }


def route_after_skill(state: TutorialAgentState) -> Literal["build_skill_material", "rag_required"]:
    return "build_skill_material" if state.get("skill_matches") else "rag_required"


def build_skill_material_node(state: TutorialAgentState) -> dict[str, Any]:
    matches = state.get("skill_matches") or []
    if not matches:
        return {
            "status": "rag_required",
            "next_action": "ragflow_retrieve",
            "trace": append_trace(state, "build_skill_material:no_match"),
        }
    materials = [build_skill_material(matches[0])]
    return {
        "materials": materials,
        "material_summaries": serialize_materials(materials),
        "status": "skill_material_ready",
        "trace": append_trace(state, "build_skill_material"),
    }


def rag_required_node(state: TutorialAgentState) -> dict[str, Any]:
    return {
        "status": "rag_required",
        "answer": "本地 Skill 未找到足够可靠的操作教程，需要进入 RAGFlow 知识库检索后再交给大模型总结。",
        "answer_mode": "rag_required",
        "next_action": "ragflow_retrieve",
        "trace": append_trace(state, "rag_required"),
    }


def llm_summarize_node_factory(llm_client: LLMClient | None = None):
    def llm_summarize_node(state: TutorialAgentState) -> dict[str, Any]:
        materials = state.get("materials") or []
        matches = state.get("skill_matches") or []
        if not materials or not matches:
            return {
                "status": "rag_required",
                "answer": "没有可用于总结的资料，需要进入 RAGFlow 知识库检索。",
                "answer_mode": "rag_required",
                "next_action": "ragflow_retrieve",
                "trace": append_trace(state, "llm_summarize:no_materials"),
            }

        client = llm_client or build_env_llm_client(
            force=bool(state.get("use_llm", False)),
            config_path=state.get("llm_config") or DEFAULT_LLM_CONFIG_PATH,
        )
        if client:
            try:
                return {
                    "status": "skill_answer",
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
            except Exception as exc:  # noqa: BLE001 - keep graph response inspectable.
                return {
                    "status": "skill_answer",
                    "answer": render_tutorial_answer(matches[0], state["question"]),
                    "answer_mode": "template_fallback",
                    "llm_error": str(exc),
                    "next_action": None,
                    "trace": append_trace(state, "llm_summarize:fallback"),
                }

        return {
            "status": "skill_answer",
            "answer": render_tutorial_answer(matches[0], state["question"]),
            "answer_mode": "template",
            "llm_error": None,
            "next_action": None,
            "trace": append_trace(state, "llm_summarize:template"),
        }

    return llm_summarize_node


def final_node(state: TutorialAgentState) -> dict[str, Any]:
    return {
        "trace": append_trace(state, "final"),
        "matches": state.get("matches", []),
        "material_summaries": state.get("material_summaries", []),
    }


def build_tutorial_graph(llm_client: LLMClient | None = None):
    builder = StateGraph(TutorialAgentState)
    builder.add_node("load_context", load_context_node)
    builder.add_node("retrieve_skill", retrieve_skill_node)
    builder.add_node("build_skill_material", build_skill_material_node)
    builder.add_node("rag_required", rag_required_node)
    builder.add_node("llm_summarize", llm_summarize_node_factory(llm_client=llm_client))
    builder.add_node("final", final_node)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "retrieve_skill")
    builder.add_conditional_edges(
        "retrieve_skill",
        route_after_skill,
        {
            "build_skill_material": "build_skill_material",
            "rag_required": "rag_required",
        },
    )
    builder.add_edge("build_skill_material", "llm_summarize")
    builder.add_edge("llm_summarize", "final")
    builder.add_edge("rag_required", "final")
    builder.add_edge("final", END)
    return builder.compile()


def ask_graph_agent(
    question: str,
    *,
    module: str | None = None,
    page: str | None = None,
    skills_root: str | Path = "skill_demo/examples",
    llm_config: str | Path = DEFAULT_LLM_CONFIG_PATH,
    use_llm: bool = False,
    llm_client: LLMClient | None = None,
) -> TutorialAgentState:
    graph = build_tutorial_graph(llm_client=llm_client)
    initial_state: TutorialAgentState = {
        "question": question,
        "module": module,
        "page": page,
        "skills_root": str(skills_root),
        "llm_config": str(llm_config),
        "use_llm": use_llm,
    }
    return graph.invoke(initial_state)


def public_result(state: TutorialAgentState) -> dict[str, Any]:
    return {
        "status": state.get("status"),
        "answer": state.get("answer"),
        "answer_mode": state.get("answer_mode"),
        "llm_error": state.get("llm_error"),
        "next_action": state.get("next_action"),
        "matches": state.get("matches", []),
        "materials": state.get("material_summaries", []),
        "trace": state.get("trace", []),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LangGraph tutorial-agent skeleton.")
    parser.add_argument("question", help="User question in natural language.")
    parser.add_argument("--skills-root", default="skill_demo/examples")
    parser.add_argument("--module", default=None)
    parser.add_argument("--page", default=None)
    parser.add_argument("--llm-config", default=str(DEFAULT_LLM_CONFIG_PATH))
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    state = ask_graph_agent(
        args.question,
        module=args.module,
        page=args.page,
        skills_root=args.skills_root,
        llm_config=args.llm_config,
        use_llm=args.use_llm,
    )
    result = public_result(state)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("answer") or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
