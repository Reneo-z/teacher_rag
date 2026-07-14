from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teacher_rag.graph_skill_agent import build_graph_skill_agent, public_result
from teacher_rag.ragflow_retriever import DEFAULT_RAGFLOW_CONFIG_PATH
from teacher_rag.skill_agent import DEFAULT_LLM_CONFIG_PATH


WEB_ROOT = ROOT / "web"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "web_chat_runs.jsonl"

ERROR_STATUSES = {
    "skill_selection_error",
    "skill_selection_unavailable",
    "ragflow_error",
    "server_error",
}
WARNING_STATUSES = {
    "context_extraction_error",
    "context_extraction_unavailable",
    "not_found",
}

STEP_START_MESSAGES = {
    "load_context": "正在初始化问答上下文",
    "load_skill_catalog": "正在加载教程 Skill 目录",
    "extract_context": "正在从问题中识别模块",
    "select_skill": "正在匹配最相关的 Skill",
    "build_skill_material": "已命中 Skill，正在读取教程资料",
    "ragflow_retrieve": "Skill 未命中，正在检索知识库",
    "build_rag_material": "正在整理知识库片段",
    "llm_summarize": "正在让大模型生成精简流程",
    "final": "正在收尾本次问答",
}

STEP_DONE_MESSAGES = {
    "load_context": "问答上下文初始化完成",
    "load_skill_catalog": "教程 Skill 目录加载完成",
    "extract_context": "模块识别完成",
    "select_skill": "Skill 匹配完成",
    "build_skill_material": "Skill 教程资料已准备好",
    "ragflow_retrieve": "知识库检索完成",
    "build_rag_material": "知识库片段已准备好",
    "llm_summarize": "精简流程生成完成",
    "final": "问答流程完成",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def write_log(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"ts": now_iso(), **record}
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def short_text(value: str | None, limit: int = 240) -> str | None:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def step_success(status: str | None, trace_event: str | None) -> bool:
    if status in ERROR_STATUSES:
        return False
    if trace_event and (":error" in trace_event or trace_event.endswith(":no_llm")):
        return False
    return True


def step_level(status: str | None, trace_event: str | None) -> str:
    if not step_success(status, trace_event):
        return "error"
    if status in WARNING_STATUSES or (trace_event and "not_found" in trace_event):
        return "warning"
    return "success"


def build_initial_state(question: str) -> dict:
    return {
        "question": question,
        "module": None,
        "page": None,
        "explicit_module": None,
        "explicit_page": None,
        "skills_root": "skill_demo/examples",
        "llm_config": str(DEFAULT_LLM_CONFIG_PATH),
        "use_llm": True,
        "ragflow_config": str(DEFAULT_RAGFLOW_CONFIG_PATH),
    }


def step_message(node_name: str, phase: str, state: dict | None = None) -> str:
    state = state or {}
    if phase == "started":
        return STEP_START_MESSAGES.get(node_name, f"正在执行 {node_name}")
    if node_name == "extract_context" and state.get("module"):
        return f"模块识别完成：{state['module']}"
    if node_name == "select_skill" and state.get("selected_skill_name"):
        confidence = state.get("selection_confidence")
        suffix = f"，置信度 {confidence:.2f}" if isinstance(confidence, float) else ""
        return f"已命中 Skill：{state['selected_skill_name']}{suffix}"
    if node_name == "ragflow_retrieve" and state.get("raw_rag_chunk_count") is not None:
        return f"知识库召回 {state['raw_rag_chunk_count']} 个片段"
    return STEP_DONE_MESSAGES.get(node_name, f"{node_name} 执行完成")


def run_agent_with_step_logs(question: str, request_id: str) -> dict:
    graph = build_graph_skill_agent()
    state: dict = build_initial_state(question)
    started = time.perf_counter()
    write_log({
        "request_id": request_id,
        "event": "graph_started",
        "current_step": "load_context",
        "question": question,
    })

    for update in graph.stream(state, stream_mode="updates"):
        for node_name, delta in update.items():
            if isinstance(delta, dict):
                state.update(delta)
            trace = state.get("trace") or []
            trace_event = trace[-1] if trace else node_name
            status = state.get("status")
            write_log({
                "request_id": request_id,
                "event": "graph_step",
                "step": node_name,
                "trace_event": trace_event,
                "status": status,
                "success": step_success(status, trace_event),
                "level": step_level(status, trace_event),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "module": state.get("module"),
                "extracted_module": state.get("extracted_module"),
                "selected_skill_name": state.get("selected_skill_name"),
                "selection_confidence": state.get("selection_confidence"),
                "answer_mode": state.get("answer_mode"),
                "raw_rag_chunk_count": state.get("raw_rag_chunk_count"),
                "llm_error": short_text(state.get("llm_error")),
                "next_action": state.get("next_action"),
            })

    result = public_result(state)
    write_log({
        "request_id": request_id,
        "event": "graph_finished",
        "status": result.get("status"),
        "success": result.get("status") not in ERROR_STATUSES,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "answer_mode": result.get("answer_mode"),
        "module": result.get("module"),
        "selected_skill_name": result.get("selected_skill_name"),
        "materials_count": len(result.get("materials") or []),
        "raw_rag_chunk_count": result.get("raw_rag_chunk_count"),
        "llm_error": short_text(result.get("llm_error")),
        "answer_preview": short_text(result.get("answer")),
        "trace": result.get("trace", []),
    })
    return result


def stream_agent_events(question: str, request_id: str):
    graph = build_graph_skill_agent()
    state: dict = build_initial_state(question)
    started = time.perf_counter()
    write_log({
        "request_id": request_id,
        "event": "graph_started",
        "current_step": "load_context",
        "question": question,
    })
    yield {
        "event": "graph_started",
        "request_id": request_id,
        "message": "正在进入问答流程",
        "elapsed_ms": 0,
    }

    for debug_event in graph.stream(state, stream_mode="debug"):
        payload = debug_event.get("payload") or {}
        node_name = payload.get("name")
        if not node_name:
            continue

        if debug_event.get("type") == "task":
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            record = {
                "request_id": request_id,
                "event": "graph_step_started",
                "step": node_name,
                "message": step_message(node_name, "started", state),
                "elapsed_ms": elapsed_ms,
            }
            write_log(record)
            yield {**record, "phase": "started"}
            continue

        if debug_event.get("type") != "task_result":
            continue

        result_delta = payload.get("result")
        if isinstance(result_delta, dict):
            state.update(result_delta)
        trace = state.get("trace") or []
        trace_event = trace[-1] if trace else node_name
        status = state.get("status")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        error = payload.get("error")
        record = {
            "request_id": request_id,
            "event": "graph_step_finished",
            "step": node_name,
            "trace_event": trace_event,
            "status": status,
            "success": error is None and step_success(status, trace_event),
            "level": "error" if error else step_level(status, trace_event),
            "message": short_text(str(error), 240) if error else step_message(node_name, "finished", state),
            "elapsed_ms": elapsed_ms,
            "module": state.get("module"),
            "extracted_module": state.get("extracted_module"),
            "selected_skill_name": state.get("selected_skill_name"),
            "selection_confidence": state.get("selection_confidence"),
            "answer_mode": state.get("answer_mode"),
            "raw_rag_chunk_count": state.get("raw_rag_chunk_count"),
            "llm_error": short_text(state.get("llm_error")),
            "next_action": state.get("next_action"),
        }
        write_log(record)
        yield {**record, "phase": "finished"}

    result = public_result(state)
    final_record = {
        "request_id": request_id,
        "event": "graph_finished",
        "status": result.get("status"),
        "success": result.get("status") not in ERROR_STATUSES,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "answer_mode": result.get("answer_mode"),
        "module": result.get("module"),
        "selected_skill_name": result.get("selected_skill_name"),
        "materials_count": len(result.get("materials") or []),
        "raw_rag_chunk_count": result.get("raw_rag_chunk_count"),
        "llm_error": short_text(result.get("llm_error")),
        "answer_preview": short_text(result.get("answer")),
        "trace": result.get("trace", []),
    }
    write_log(final_record)
    result["request_id"] = request_id
    yield {"event": "final", "request_id": request_id, "result": result}


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/tutorial-agent/chat/stream":
            self._handle_stream_chat()
            return
        if path != "/api/tutorial-agent/chat":
            self.send_error(404)
            return

        request_id = uuid.uuid4().hex[:12]
        request_started = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(payload.get("question") or "").strip()
            write_log({
                "request_id": request_id,
                "event": "request_received",
                "path": self.path,
                "client": self.client_address[0] if self.client_address else None,
                "question": question,
            })
            if not question:
                response = {"status": "bad_request", "answer": "请输入问题。"}
                write_log({
                    "request_id": request_id,
                    "event": "request_finished",
                    "status": "bad_request",
                    "success": False,
                    "elapsed_ms": round((time.perf_counter() - request_started) * 1000, 2),
                })
                self._send_json(response, status=400)
                return

            result = run_agent_with_step_logs(question, request_id)
            result["request_id"] = request_id
            self._send_json(result)
            write_log({
                "request_id": request_id,
                "event": "response_sent",
                "http_status": 200,
                "elapsed_ms": round((time.perf_counter() - request_started) * 1000, 2),
            })
        except Exception as exc:  # noqa: BLE001 - demo endpoint returns inspectable error.
            write_log({
                "request_id": request_id,
                "event": "request_failed",
                "status": "server_error",
                "success": False,
                "elapsed_ms": round((time.perf_counter() - request_started) * 1000, 2),
                "error": short_text(str(exc), limit=800),
            })
            self._send_json(
                {
                    "status": "server_error",
                    "answer": "服务暂时不可用，请稍后重试。",
                    "error": str(exc),
                    "request_id": request_id,
                },
                status=500,
            )

    def _handle_stream_chat(self) -> None:
        request_id = uuid.uuid4().hex[:12]
        request_started = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(payload.get("question") or "").strip()
            write_log({
                "request_id": request_id,
                "event": "request_received",
                "path": self.path,
                "client": self.client_address[0] if self.client_address else None,
                "question": question,
                "stream": True,
            })
            if not question:
                self._send_json({"status": "bad_request", "answer": "请输入问题。"}, status=400)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            for event in stream_agent_events(question, request_id):
                event_name = str(event.get("event") or "message")
                self._send_sse(event_name, event)
            self._send_sse("done", {"request_id": request_id})
            write_log({
                "request_id": request_id,
                "event": "response_sent",
                "http_status": 200,
                "stream": True,
                "elapsed_ms": round((time.perf_counter() - request_started) * 1000, 2),
            })
        except Exception as exc:  # noqa: BLE001 - stream endpoint returns inspectable error.
            write_log({
                "request_id": request_id,
                "event": "request_failed",
                "status": "server_error",
                "success": False,
                "stream": True,
                "elapsed_ms": round((time.perf_counter() - request_started) * 1000, 2),
                "error": short_text(str(exc), limit=800),
            })
            try:
                self._send_sse(
                    "error",
                    {
                        "request_id": request_id,
                        "status": "server_error",
                        "answer": "服务暂时不可用，请稍后重试。",
                        "error": str(exc),
                    },
                )
            except Exception:
                pass

    def _send_sse(self, event_name: str, payload: dict) -> None:
        body = (
            f"event: {event_name}\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        ).encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    host = "127.0.0.1"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Tutorial agent web demo: http://{host}:{port}")
    print(f"Execution log file: {LOG_FILE}")
    print("Static asset version: 20260521-final")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
