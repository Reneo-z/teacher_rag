const API_URL = window.TUTOR_AGENT_API_URL || "/api/tutorial-agent/chat";
const STREAM_API_URL =
  window.TUTOR_AGENT_STREAM_API_URL ||
  (API_URL.endsWith("/chat") ? `${API_URL}/stream` : "/api/tutorial-agent/chat/stream");

const widget = document.querySelector(".assistant-widget");
const fab = document.querySelector(".assistant-fab");
const panel = document.querySelector(".assistant-panel");
const closeButton = document.querySelector("#closeButton");
const minimizeButton = document.querySelector("#minimizeButton");
const newChatButton = document.querySelector("#newChatButton");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#chatInput");
const messageList = document.querySelector("#messageList");
const emptyState = document.querySelector("#emptyState");
const subtitle = document.querySelector("#assistantSubtitle");

const state = {
  sessionId: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
  messages: [],
  loading: false,
};


function openPanel() {
  panel.hidden = false;
  fab.hidden = true;
  fab.setAttribute("aria-expanded", "true");
  setTimeout(() => input.focus(), 0);
}

function closePanel() {
  panel.hidden = true;
  fab.hidden = false;
  fab.setAttribute("aria-expanded", "false");
}

function resetChat() {
  state.sessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
  state.messages = [];
  messageList.innerHTML = "";
  messageList.appendChild(emptyState);
  emptyState.hidden = false;
  subtitle.textContent = "新对话";
  input.value = "";
  input.focus();
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatAnswer(value) {
  const safe = escapeHtml(value || "");
  return safe
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/^### (.*)$/gm, "<strong>$1</strong>")
    .replace(/^\d+\. /gm, (match) => `\n${match}`)
    .trim();
}

function sourceText(meta) {
  if (!meta) return "";
  if (meta.status === "skill_answer") {
    return meta.selected_skill_name
      ? `来源：Skill · ${meta.selected_skill_name}`
      : "来源：Skill";
  }
  if (meta.status === "rag_answer") {
    const used = Array.isArray(meta.materials) ? meta.materials.length : 0;
    const raw = meta.raw_rag_chunk_count;
    return raw ? `来源：知识库 · 使用 ${used} / 召回 ${raw} 个片段` : `来源：知识库 · ${used} 个片段`;
  }
  return "";
}

function appendMessage(role, content, meta = null) {
  emptyState.hidden = true;
  if (emptyState.parentElement === messageList) {
    emptyState.remove();
  }

  const item = document.createElement("div");
  item.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = role === "assistant" ? formatAnswer(content) : escapeHtml(content);

  const source = sourceText(meta);
  if (source) {
    const footer = document.createElement("div");
    footer.className = "source-line";
    footer.textContent = source;
    bubble.appendChild(footer);
  }

  item.appendChild(bubble);
  messageList.appendChild(item);
  messageList.scrollTop = messageList.scrollHeight;
  return item;
}

function appendLoading(message = "正在提交问题") {
  emptyState.hidden = true;
  if (emptyState.parentElement === messageList) {
    emptyState.remove();
  }

  const item = document.createElement("div");
  item.className = "message assistant pending";

  const bubble = document.createElement("div");
  bubble.className = "bubble thinking-bubble";

  const row = document.createElement("div");
  row.className = "thinking-row";

  const text = document.createElement("span");
  text.className = "thinking-text";
  text.textContent = message;

  const dots = document.createElement("span");
  dots.className = "loading-dots";
  for (let index = 0; index < 3; index += 1) {
    dots.appendChild(document.createElement("span"));
  }

  row.append(text, dots);
  bubble.appendChild(row);
  item.appendChild(bubble);
  messageList.appendChild(item);
  messageList.scrollTop = messageList.scrollHeight;

  return {
    item,
    update(nextMessage) {
      if (nextMessage) {
        text.textContent = nextMessage;
        messageList.scrollTop = messageList.scrollHeight;
      }
    },
    remove() {
      item.remove();
    },
  };
}

function appendAssistantStreaming(content, meta = null) {
  emptyState.hidden = true;
  if (emptyState.parentElement === messageList) {
    emptyState.remove();
  }

  const item = document.createElement("div");
  item.className = "message assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble streaming";
  item.appendChild(bubble);
  messageList.appendChild(item);

  const answer = content || "我暂时没有找到可靠答案。";
  let cursor = 0;
  const step = Math.max(1, Math.ceil(answer.length / 80));

  return new Promise((resolve) => {
    const render = () => {
      cursor = Math.min(cursor + step, answer.length);
      bubble.innerHTML = `${formatAnswer(answer.slice(0, cursor))}<span class="stream-cursor"></span>`;
      messageList.scrollTop = messageList.scrollHeight;

      if (cursor < answer.length) {
        window.setTimeout(render, 18);
        return;
      }

      bubble.innerHTML = formatAnswer(answer);
      const source = sourceText(meta);
      if (source) {
        const footer = document.createElement("div");
        footer.className = "source-line";
        footer.textContent = source;
        bubble.appendChild(footer);
      }
      messageList.scrollTop = messageList.scrollHeight;
      resolve(item);
    };

    render();
  });
}

async function askAgent(question) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      question,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function parseSseBlock(block) {
  const lines = block.split("\n");
  let event = "message";
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (!dataLines.length) return null;
  return { event, data: JSON.parse(dataLines.join("\n")) };
}

async function askAgentStream(question, onProgress) {
  const response = await fetch(STREAM_API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      question,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  if (!response.body) {
    throw new Error("当前浏览器不支持流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalResult = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      const parsed = parseSseBlock(block.trim());
      if (!parsed) continue;

      if (parsed.event === "graph_step_started" || parsed.event === "graph_step_finished") {
        onProgress(parsed.data);
      } else if (parsed.event === "graph_started") {
        onProgress(parsed.data);
      } else if (parsed.event === "final") {
        finalResult = parsed.data.result;
        return finalResult;
      } else if (parsed.event === "done" && finalResult) {
        return finalResult;
      } else if (parsed.event === "error") {
        throw new Error(parsed.data.error || parsed.data.answer || "服务暂时不可用");
      }
    }
  }

  if (!finalResult) {
    throw new Error("问答流程没有返回最终结果");
  }
  return finalResult;
}

async function submitQuestion(question) {
  const normalized = question.trim();
  if (!normalized || state.loading) return;

  state.loading = true;
  input.value = "";
  input.style.height = "auto";
  appendMessage("user", normalized);
  const loading = appendLoading();
  subtitle.textContent = "思考中";
  form.querySelector("button").disabled = true;

  try {
    const result = await askAgentStream(normalized, (progress) => {
      if (progress.message) {
        loading.update(progress.message);
      }
    });
    loading.remove();
    await appendAssistantStreaming(result.answer || "我暂时没有找到可靠答案。", result);
    subtitle.textContent = result.module ? `${result.module}` : "新对话";
  } catch (error) {
    loading.remove();
    appendMessage("assistant", `请求失败：${error.message || "请稍后重试"}`, {
      status: "error",
    });
    subtitle.textContent = "请求失败";
  } finally {
    state.loading = false;
    form.querySelector("button").disabled = false;
    input.focus();
  }
}

fab.addEventListener("click", openPanel);
closeButton.addEventListener("click", closePanel);
minimizeButton.addEventListener("click", closePanel);
newChatButton.addEventListener("click", resetChat);

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(input.value);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".prompt-row button").forEach((button) => {
  button.addEventListener("click", () => {
    openPanel();
    submitQuestion(button.textContent || "");
  });
});

window.addEventListener("message", (event) => {
  if (!event.data || event.data.type !== "TUTOR_AGENT_OPEN") return;
  openPanel();
});
