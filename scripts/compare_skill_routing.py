from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teacher_rag.graph_agent import ask_graph_agent, public_result as public_keyword_graph_result
from teacher_rag.graph_skill_agent import ask_graph_skill_agent, public_result as public_llm_graph_result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare keyword-router graph with LangGraph+LLM Skill router.")
    parser.add_argument("question")
    parser.add_argument("--module", default=None)
    parser.add_argument("--page", default=None)
    parser.add_argument("--skills-root", default="skill_demo/examples")
    parser.add_argument("--llm-config", default="config/llm.local.json")
    return parser


def summarize(result: dict) -> dict:
    return {
        "status": result.get("status"),
        "answer_mode": result.get("answer_mode"),
        "selected_skill_name": result.get("selected_skill_name"),
        "matched_skill_name": result.get("matches", [{}])[0].get("name") if result.get("matches") else None,
        "next_action": result.get("next_action"),
        "trace": result.get("trace"),
        "answer": result.get("answer"),
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    keyword_state = ask_graph_agent(
        args.question,
        module=args.module,
        page=args.page,
        skills_root=args.skills_root,
        llm_config=args.llm_config,
        use_llm=True,
    )
    llm_state = ask_graph_skill_agent(
        args.question,
        module=args.module,
        page=args.page,
        skills_root=args.skills_root,
        llm_config=args.llm_config,
        use_llm=True,
    )
    output = {
        "question": args.question,
        "module": args.module,
        "keyword_router_graph": summarize(public_keyword_graph_result(keyword_state)),
        "llm_skill_router_graph": summarize(public_llm_graph_result(llm_state)),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
