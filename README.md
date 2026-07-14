# teacher_rag

智能教程 Agent 原型项目。

## 第一阶段：Skill 加载与问答验证

当前已跑通最小闭环：

1. 从 Skill 目录扫描 `SKILL.md`。
2. 解析 frontmatter、概览、前置条件、操作步骤、验证方式和常见异常。
3. 根据用户问题、当前模块、Skill 描述、关键词和典型问法做本地匹配。
4. 命中 Skill 时先构造检索资料包，再交给大模型总结精简操作流程。
5. 未命中 Skill 时返回 `no_skill_match`，后续交给 RAGFlow 兜底。

默认使用示例目录：

```bash
python3 scripts/ask_skill_agent.py "怎么建巡检任务？" --module 任务管理
```

输出结构化结果：

```bash
python3 scripts/ask_skill_agent.py "派单怎么操作？" --module 任务管理 --json
```

只测试本地匹配和模板兜底时，不需要模型配置。需要让大模型根据用户问题聚焦总结时，复制配置模板：

```bash
cp config/llm.example.json config/llm.local.json
```

然后编辑 `config/llm.local.json`：

```json
{
  "enabled": true,
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o-mini",
  "api_key": "你的开发阶段 API Key",
  "timeout_seconds": 60,
  "temperature": 0.2
}
```

运行：

```bash
python3 scripts/ask_skill_agent.py "创建巡检任务时，如何指派执行人？" --module 任务管理 --use-llm
```

交付阶段切到本地部署模型时，只需要把 `config/llm.local.json` 改成本地 OpenAI-compatible `/chat/completions` 地址：

```json
{
  "enabled": true,
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "本地模型名称",
  "api_key": "",
  "timeout_seconds": 60,
  "temperature": 0.2
}
```

`config/llm.local.json` 已被 `.gitignore` 忽略，不要把真实 API Key 写入 `config/llm.example.json`。当前 LLM 层的职责是：Skill 或后续 RAGFlow 只提供资料，最终回答由模型基于“用户问题 + 检索资料”抽取并总结。

运行验证：

```bash
python3 scripts/verify_skill_agent.py
python3 -m unittest discover -s tests -v
```


## 第二阶段：LangGraph 编排骨架

当前已把第一阶段能力接入 LangGraph：

```text
load_context
  -> retrieve_skill
  -> build_skill_material 或 rag_required
  -> llm_summarize
  -> final
```

第一次运行前安装依赖到项目虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

运行 Skill 命中 + LLM 总结路径：

```bash
.venv/bin/python scripts/ask_graph_agent.py "创建巡检任务时，如何指派执行人？" --module 任务管理 --use-llm --json
```

运行 Skill 未命中路径：

```bash
.venv/bin/python scripts/ask_graph_agent.py "怎么修改个人头像？" --module 个人中心 --json
```

未命中时会返回：

```json
{
  "status": "rag_required",
  "next_action": "ragflow_retrieve"
}
```

这就是第三阶段接入 RAGFlow 的预留分支。


## LangGraph + LLM Skill 路由版

保留旧版关键词路由图 `scripts/ask_graph_agent.py`，新增一个并行版本：

```bash
.venv/bin/python scripts/ask_graph_skill_agent.py "创建巡检任务时，如何指派执行人？" --module 任务管理 --json
```

这个版本的流程是：

```text
load_context
  -> load_skill_catalog
  -> select_skill  # LLM 根据问题和 Skill catalog 选择 Skill
  -> 如果命中 Skill：build_skill_material -> llm_summarize -> final
  -> 如果未命中 Skill：ragflow_retrieve -> build_rag_material -> llm_summarize -> final
```

对比两种路由方式：

```bash
.venv/bin/python scripts/compare_skill_routing.py "创建巡检任务时，如何指派执行人？" --module 任务管理
```

后续可用这个脚本批量统计旧版关键词路由和新版 LLM Skill 路由的命中准确率。




### 自然语言模块抽取

现在 `ask_graph_skill_agent.py` 不再要求必须显式传 `--module`。如果用户问题中包含模块信息，图里的 `extract_context` 节点会先用大模型抽取模块和页面，再继续 Skill/RAGFlow 路由。

示例：

```bash
.venv/bin/python scripts/ask_graph_skill_agent.py "我在任务管理模块中想知道在创建巡检任务时，如何指派执行人？" --json
```

预期输出中会包含：

```json
{
  "module": "任务管理",
  "extracted_module": "任务管理",
  "context_extraction_confidence": 0.95
}
```

如果命令行显式传了 `--module`，显式参数优先，抽取节点不会覆盖它：

```bash
.venv/bin/python scripts/ask_graph_skill_agent.py "创建巡检任务时，如何指派执行人？" --module 任务管理 --json
```

### Skill + RAGFlow 联合测试

Skill 命中时不会走 RAGFlow：

```bash
.venv/bin/python scripts/ask_graph_skill_agent.py "创建巡检任务时，如何指派执行人？" --module 任务管理 --json
```

RAGFlow 兜底路径：

```bash
.venv/bin/python scripts/ask_graph_skill_agent.py "用户问答流程是什么？" --module 任务管理 --json
```

预期 RAGFlow 路径返回：

```json
{
  "status": "rag_answer",
  "answer_mode": "llm",
  "raw_rag_chunk_count": 10,
  "materials": [
    {"source_type": "ragflow"}
  ]
}
```

其中 `raw_rag_chunk_count` 是 RAGFlow 原始召回数量，`materials` 是经过 `max_chunks_for_llm`、`min_similarity_for_llm`、`max_chunk_chars` 筛选后传给大模型的 chunk。

## 独立 RAGFlow Retriever 测试

先复制模板并填写本地配置：

```bash
cp config/ragflow.example.json config/ragflow.local.json
```

`config/ragflow.local.json` 示例：

```json
{
  "enabled": true,
  "base_url": "http://127.0.0.1:9380",
  "api_key": "你的 RAGFlow API Key",
  "default_dataset_ids": ["你的 dataset_id"],
  "module_dataset_map": {
    "任务管理": ["你的 dataset_id"]
  },
  "retrieval": {
    "page": 1,
    "page_size": 5,
    "similarity_threshold": 0.2,
    "vector_similarity_weight": 0.3,
    "top_k": 1024,
    "keyword": true
  },
  "timeout_seconds": 60
}
```

运行独立检索：

```bash
.venv/bin/python scripts/test_ragflow_retriever.py "执行人列表为空怎么办？" --module 任务管理
```

结构化输出：

```bash
.venv/bin/python scripts/test_ragflow_retriever.py "执行人列表为空怎么办？" --module 任务管理 --json
```

查看最终准备传给 LLM 的 chunk：

```bash
.venv/bin/python scripts/test_ragflow_retriever.py "执行人列表为空怎么办？" --module 任务管理 --for-llm --json
```

临时指定 dataset：

```bash
.venv/bin/python scripts/test_ragflow_retriever.py "执行人列表为空怎么办？" --dataset-id your_dataset_id --json
```

`config/ragflow.local.json` 已被 `.gitignore` 忽略，不要把真实 API Key 写入 `config/ragflow.example.json`。


## 最小 Web 问答页面原型

已提供一个无需前端构建工具的静态 Web 原型，并用 Python 标准库提供 demo API：

```bash
.venv/bin/python scripts/run_web_demo.py 8787
```

访问：

```text
http://127.0.0.1:8787
```

页面包含：

- 右下角悬浮机器人按钮
- 白色聊天面板
- 新对话、最小化、关闭
- 输入框、发送按钮、loading、错误态
- Skill / RAGFlow 来源提示
- `/api/tutorial-agent/chat` demo 接口

API 请求格式：

```json
{
  "session_id": "optional-session-id",
  "question": "我在任务管理模块中想知道在创建巡检任务时，如何指派执行人？"
}
```

该 demo 接口会调用现有 `ask_graph_skill_agent(question)`，不要求前端显式传 `module`。

## 关键文件

- `src/teacher_rag/skill_agent.py`：Skill loader、router、资料包构造、LLM 调用、模板兜底。
- `src/teacher_rag/graph_agent.py`：LangGraph 编排骨架，沿用旧关键词 SkillRouter。
- `src/teacher_rag/graph_skill_agent.py`：LangGraph + LLM Skill catalog 选择流程。
- `src/teacher_rag/ragflow_retriever.py`：独立 RAGFlow Retriever，暂未接入 LangGraph。
- `tests/test_skill_agent.py`：第一阶段单元测试。
- `scripts/verify_skill_agent.py`：命中/未命中闭环验证。
- `scripts/ask_skill_agent.py`：本地问答 CLI。
- `scripts/ask_graph_agent.py`：LangGraph 编排 CLI。
- `scripts/ask_graph_skill_agent.py`：LangGraph + LLM Skill 路由 CLI。
- `scripts/compare_skill_routing.py`：旧版关键词路由与新版 LLM 路由对比 CLI。
- `scripts/test_ragflow_retriever.py`：独立 RAGFlow chunk 检索测试 CLI。
- `scripts/run_web_demo.py`：最小 Web 问答页面和 demo API 服务。
- `web/`：静态问答组件原型。
- `skill_demo/`：Skill 模板、标准说明和示例。
