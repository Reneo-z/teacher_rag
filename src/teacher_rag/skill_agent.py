"""Minimal Skill loading and tutorial-answer verification loop.

This module intentionally has no third-party runtime dependency. It can load
LangChain Deep Agents style Skill directories, route a user's question to the
best matching Skill, and render a natural-language tutorial answer from the
Skill body.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol


SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
STEP_RE = re.compile(r"^###\s+步骤\s*([0-9一二三四五六七八九十N]+)[：:]\s*(.+?)\s*$", re.MULTILINE)
QUOTE_RE = re.compile(r"[“\"]([^”\"]+)[”\"]")
DEFAULT_LLM_CONFIG_PATH = Path("config/llm.local.json")


@dataclass(frozen=True)
class SkillStep:
    index: str
    title: str
    action: str
    location: str = ""
    note: str = ""


@dataclass(frozen=True)
class Skill:
    path: Path
    name: str
    description: str
    metadata: dict[str, Any]
    title: str
    overview: str
    scenarios: list[str]
    prerequisites: list[str]
    steps: list[SkillStep]
    verification: list[str]
    exceptions: list[dict[str, str]]
    related: list[str]
    raw_frontmatter: dict[str, Any] = field(repr=False)
    raw_body: str = field(repr=False)

    @property
    def display_name(self) -> str:
        return str(self.metadata.get("display_name") or self.title or self.name)

    @property
    def module(self) -> str:
        return str(self.metadata.get("module") or "")

    @property
    def system(self) -> str:
        return str(self.metadata.get("system") or "")

    @property
    def keywords(self) -> list[str]:
        values = self.metadata.get("keywords") or []
        return [str(item) for item in values if str(item).strip()]


@dataclass(frozen=True)
class SkillMatch:
    skill: Skill
    score: float
    reasons: list[str]


@dataclass(frozen=True)
class RetrievedMaterial:
    """A normalized evidence bundle from Skill or future RAGFlow retrieval."""

    source_type: str
    title: str
    content: str
    metadata: dict[str, Any]


class LLMClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return a model-generated answer."""


@dataclass(frozen=True)
class OpenAICompatibleChatLLM:
    """Small OpenAI-compatible chat-completions client.

    Works with OpenAI-compatible local deployments by changing ``base_url`` and
    ``model``. It avoids SDK dependencies for this prototype.
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: int = 60
    temperature: float = 0.2

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleChatLLM":
        return cls(
            base_url=str(config.get("base_url") or "https://api.openai.com/v1"),
            model=str(config.get("model") or "gpt-4o-mini"),
            api_key=normalize_optional_str(config.get("api_key")),
            timeout_seconds=int(config.get("timeout_seconds") or 60),
            temperature=float(config.get("temperature") or 0.2),
        )

    @classmethod
    def from_env(cls) -> "OpenAICompatibleChatLLM":
        return cls.from_config(
            {
                "base_url": os.getenv("TUTOR_AGENT_LLM_BASE_URL", "https://api.openai.com/v1"),
                "model": os.getenv("TUTOR_AGENT_LLM_MODEL", "gpt-4o-mini"),
                "api_key": os.getenv("TUTOR_AGENT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
                "timeout_seconds": os.getenv("TUTOR_AGENT_LLM_TIMEOUT", "60"),
                "temperature": os.getenv("TUTOR_AGENT_LLM_TEMPERATURE", "0.2"),
            }
        )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        parsed = json.loads(raw)
        try:
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {raw[:500]}") from exc
        return str(content).strip()


def load_skills(root: str | Path) -> list[Skill]:
    """Load every Skill directory under ``root``.

    A valid Skill is any directory containing a ``SKILL.md`` file.
    """

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Skill root does not exist: {root_path}")

    skill_files = sorted(root_path.rglob("SKILL.md"))
    return [load_skill(path) for path in skill_files]


def load_skill(path: str | Path) -> Skill:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(text)
    meta = dict(frontmatter.get("metadata") or {})

    return Skill(
        path=path,
        name=str(frontmatter.get("name") or path.parent.name),
        description=str(frontmatter.get("description") or ""),
        metadata=meta,
        title=parse_title(body),
        overview=parse_section_text(body, "概览"),
        scenarios=parse_bullets(parse_section_text(body, "适用场景")),
        prerequisites=parse_checklist(parse_section_text(body, "前置条件")),
        steps=parse_steps(body),
        verification=parse_numbered_list(parse_section_text(body, "验证方式")),
        exceptions=parse_exception_table(parse_section_text(body, "常见异常 / 错误处理")),
        related=parse_bullets(parse_section_text(body, "关联资料")),
        raw_frontmatter=frontmatter,
        raw_body=body,
    )


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    frontmatter_text = text[4:end].strip("\n")
    body = text[end + len("\n---") :].lstrip("\n")
    return parse_simple_yaml(frontmatter_text), body


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the subset of YAML used by the project Skill template.

    The parser supports top-level scalar keys, ``|`` block scalars, a nested
    ``metadata`` dictionary, simple lists, and inline empty arrays.
    """

    result: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if line.startswith(" "):
            i += 1
            continue
        key, value = split_key_value(line)
        if value == "|":
            block, i = collect_block(lines, i + 1, indent=2)
            result[key] = block.rstrip()
            continue
        if value == "":
            nested, i = collect_mapping(lines, i + 1)
            result[key] = nested
            continue
        result[key] = parse_scalar(value)
        i += 1
    return result


def collect_mapping(lines: list[str], start: int) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if not line.startswith("  "):
            break
        stripped = line[2:]
        key, value = split_key_value(stripped)
        if value == "":
            values: list[Any] = []
            i += 1
            while i < len(lines) and lines[i].startswith("    - "):
                values.append(parse_scalar(lines[i][6:].strip()))
                i += 1
            mapping[key] = values
            continue
        if value == "|":
            block, i = collect_block(lines, i + 1, indent=4)
            mapping[key] = block.rstrip()
            continue
        mapping[key] = parse_scalar(value)
        i += 1
    return mapping, i


def collect_block(lines: list[str], start: int, indent: int) -> tuple[str, int]:
    prefix = " " * indent
    block: list[str] = []
    i = start
    while i < len(lines):
        if lines[i].strip() and not lines[i].startswith(prefix):
            break
        block.append(lines[i][indent:] if lines[i].startswith(prefix) else "")
        i += 1
    return "\n".join(block), i


def split_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        return line.strip(), ""
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "[]":
        return []
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def parse_title(body: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_section_text(body: str, heading: str) -> str:
    matches = list(SECTION_RE.finditer(body))
    for idx, match in enumerate(matches):
        if match.group(1).strip() != heading:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        return body[start:end].strip()
    return ""


def parse_bullets(text: str) -> list[str]:
    return [clean_markdown(line[2:]) for line in text.splitlines() if line.startswith("- ")]


def parse_checklist(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        if line.startswith("- [ ] "):
            items.append(clean_markdown(line[6:]))
        elif line.startswith("- "):
            items.append(clean_markdown(line[2:]))
    return items


def parse_numbered_list(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\d+\.\s+(.+)$", line.strip())
        if match:
            items.append(clean_markdown(match.group(1)))
    return items


def parse_steps(body: str) -> list[SkillStep]:
    matches = list(STEP_RE.finditer(body))
    steps: list[SkillStep] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        chunk = body[start:end]
        steps.append(
            SkillStep(
                index=match.group(1).strip(),
                title=match.group(2).strip(),
                action=parse_labeled_value(chunk, "操作"),
                location=parse_labeled_value(chunk, "界面位置描述"),
                note=parse_labeled_value(chunk, "说明"),
            )
        )
    return steps


def parse_labeled_value(text: str, label: str) -> str:
    pattern = rf"^-\s+\*\*{re.escape(label)}\*\*：(.+?)\s*$"
    match = re.search(pattern, text, re.MULTILINE)
    return clean_markdown(match.group(1)) if match else ""


def parse_exception_table(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped or "现象" in stripped:
            continue
        cells = [clean_markdown(cell) for cell in stripped.strip("|").split("|")]
        if len(cells) >= 3:
            rows.append({"phenomenon": cells[0], "cause": cells[1], "solution": cells[2]})
    return rows


def clean_markdown(value: str) -> str:
    return re.sub(r"`([^`]+)`", r"\1", value).strip()


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.startswith("<"):
        return None
    return normalized


def load_llm_config(path: str | Path = DEFAULT_LLM_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"LLM config must be a JSON object: {config_path}")
    return config


class SkillRouter:
    def __init__(self, skills: list[Skill], threshold: float = 3.0) -> None:
        self.skills = skills
        self.threshold = threshold

    def match(
        self,
        question: str,
        *,
        module: str | None = None,
        page: str | None = None,
        top_k: int = 3,
    ) -> list[SkillMatch]:
        matches = [self._score(skill, question, module=module, page=page) for skill in self.skills]
        matches = [match for match in matches if match.score >= self.threshold]
        return sorted(matches, key=lambda item: item.score, reverse=True)[:top_k]

    def _score(
        self,
        skill: Skill,
        question: str,
        *,
        module: str | None,
        page: str | None,
    ) -> SkillMatch:
        question_norm = normalize(question)
        reasons: list[str] = []
        score = 0.0

        if module and any_contains(module, [skill.module, skill.system, skill.name, skill.display_name]):
            score += 2.5
            reasons.append(f"当前模块与 Skill 模块相关：{skill.module or skill.system}")

        if page and any_contains(page, [skill.module, skill.system, skill.display_name]):
            score += 1.0
            reasons.append("当前页面上下文与 Skill 相关")

        for keyword in skill.keywords:
            if contains_either(question_norm, normalize(keyword)):
                score += 3.0
                reasons.append(f"命中关键词：{keyword}")

        for phrase in quoted_phrases(skill.description):
            if contains_either(question_norm, normalize(phrase)):
                score += 3.5
                reasons.append(f"命中典型问法：{phrase}")

        for term in [skill.name, skill.display_name, skill.module, skill.system]:
            term_norm = normalize(term)
            if term_norm and contains_either(question_norm, term_norm):
                score += 1.5
                reasons.append(f"命中名称或模块：{term}")

        semantic_text = " ".join([skill.description, skill.overview, " ".join(skill.scenarios)])
        similarity = SequenceMatcher(None, question_norm, normalize(semantic_text)).ratio()
        if similarity >= 0.18:
            score += similarity * 3
            reasons.append(f"语义相似度 {similarity:.2f}")

        return SkillMatch(skill=skill, score=round(score, 3), reasons=dedupe(reasons))


def quoted_phrases(text: str) -> list[str]:
    return QUOTE_RE.findall(text)


def normalize(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", "", value.lower())


def contains_either(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left in right or right in left


def any_contains(value: str, candidates: list[str]) -> bool:
    value_norm = normalize(value)
    return any(contains_either(value_norm, normalize(candidate)) for candidate in candidates)


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def render_tutorial_answer(match: SkillMatch, question: str) -> str:
    skill = match.skill
    lines: list[str] = [
        f"可以，这个问题可以使用「{skill.display_name}」教程。",
        "",
    ]

    if skill.overview:
        lines.extend([skill.overview, ""])

    if skill.prerequisites:
        lines.append("开始前请先确认：")
        lines.extend(f"- {item}" for item in skill.prerequisites)
        lines.append("")

    if skill.steps:
        lines.append("操作步骤：")
        for idx, step in enumerate(skill.steps, start=1):
            location = f"（位置：{step.location}）" if step.location else ""
            lines.append(f"{idx}. {step.title}：{step.action}{location}")
            if step.note:
                lines.append(f"   说明：{step.note}")
        lines.append("")

    if skill.verification:
        lines.append("完成后这样验证：")
        lines.extend(f"{idx}. {item}" for idx, item in enumerate(skill.verification, start=1))
        lines.append("")

    if skill.exceptions:
        lines.append("如果遇到常见问题：")
        for item in skill.exceptions[:3]:
            lines.append(
                f"- {item['phenomenon']}：通常是{item['cause']}，可以{item['solution']}。"
            )
        lines.append("")

    lines.append(f"匹配置信度：{match.score}；依据：{'；'.join(match.reasons[:4])}")
    return "\n".join(lines).strip()


def build_skill_material(match: SkillMatch) -> RetrievedMaterial:
    skill = match.skill
    lines: list[str] = [
        f"Skill 名称：{skill.display_name}",
        f"Skill ID：{skill.name}",
        f"所属系统：{skill.system}",
        f"所属模块：{skill.module}",
        "",
        "概览：",
        skill.overview,
        "",
        "适用场景：",
        *[f"- {item}" for item in skill.scenarios],
        "",
        "前置条件：",
        *[f"- {item}" for item in skill.prerequisites],
        "",
        "操作步骤：",
    ]
    for idx, step in enumerate(skill.steps, start=1):
        lines.append(f"{idx}. {step.title}")
        lines.append(f"   操作：{step.action}")
        if step.location:
            lines.append(f"   界面位置：{step.location}")
        if step.note:
            lines.append(f"   说明：{step.note}")

    if skill.verification:
        lines.extend(["", "验证方式："])
        lines.extend(f"- {item}" for item in skill.verification)

    if skill.exceptions:
        lines.extend(["", "常见异常："])
        lines.extend(
            f"- 现象：{item['phenomenon']}；原因：{item['cause']}；解决办法：{item['solution']}"
            for item in skill.exceptions
        )

    return RetrievedMaterial(
        source_type="skill",
        title=skill.display_name,
        content="\n".join(line for line in lines if line is not None).strip(),
        metadata={
            "skill_name": skill.name,
            "path": str(skill.path),
            "score": match.score,
            "reasons": match.reasons,
        },
    )


def generate_llm_tutorial_answer(
    *,
    question: str,
    materials: list[RetrievedMaterial],
    llm_client: LLMClient,
) -> str:
    system_prompt = (
        "你是一个嵌入 Web 客户端的智能教程 Agent。你的任务是根据用户问题和检索资料，"
        "总结出精简、可执行的操作流程。必须只基于提供的资料回答，不能编造页面、按钮或步骤。"
        "如果用户只问某个子步骤，只输出和该子步骤直接相关的流程，不要把完整大流程全部复述。"
        "输出要简洁，优先使用编号步骤；必要时补充前置条件和验证方式。"
    )
    material_text = "\n\n".join(
        f"【资料 {idx}】\n来源类型：{item.source_type}\n标题：{item.title}\n内容：\n{item.content}"
        for idx, item in enumerate(materials, start=1)
    )
    user_prompt = (
        f"用户问题：{question}\n\n"
        f"检索到的资料：\n{material_text}\n\n"
        "请根据用户问题，从资料中抽取最相关内容，输出精简操作教程。"
        "如果资料不足以回答，请明确说明缺少哪些信息。"
    )
    return llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)


def build_env_llm_client(
    force: bool = False,
    config_path: str | Path = DEFAULT_LLM_CONFIG_PATH,
) -> LLMClient | None:
    config = load_llm_config(config_path)
    enabled = bool(config.get("enabled")) or os.getenv("TUTOR_AGENT_USE_LLM", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not force and not enabled:
        return None

    if config:
        return OpenAICompatibleChatLLM.from_config(config)
    return OpenAICompatibleChatLLM.from_env()


def answer_question(
    question: str,
    skills_root: str | Path,
    *,
    module: str | None = None,
    page: str | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    skills = load_skills(skills_root)
    router = SkillRouter(skills)
    matches = router.match(question, module=module, page=page)
    if not matches:
        return {
            "status": "no_skill_match",
            "answer": "我暂时没有在本地 Skill 中找到足够可靠的操作教程。下一步应交给 RAGFlow 知识库兜底检索。",
            "answer_mode": "none",
            "matches": [],
            "materials": [],
        }
    best = matches[0]
    materials = [build_skill_material(best)]
    answer_mode = "template"
    llm_error = None
    if llm_client:
        try:
            answer = generate_llm_tutorial_answer(
                question=question,
                materials=materials,
                llm_client=llm_client,
            )
            answer_mode = "llm"
        except Exception as exc:  # noqa: BLE001 - surface LLM failure in response metadata.
            answer = render_tutorial_answer(best, question)
            llm_error = str(exc)
    else:
        answer = render_tutorial_answer(best, question)

    return {
        "status": "skill_answer",
        "answer": answer,
        "answer_mode": answer_mode,
        "llm_error": llm_error,
        "materials": [
            {
                "source_type": item.source_type,
                "title": item.title,
                "metadata": item.metadata,
            }
            for item in materials
        ],
        "matches": [
            {
                "name": item.skill.name,
                "display_name": item.skill.display_name,
                "score": item.score,
                "reasons": item.reasons,
                "path": str(item.skill.path),
            }
            for item in matches
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify local tutorial Skill question answering.")
    parser.add_argument("question", help="User question in natural language.")
    parser.add_argument(
        "--skills-root",
        default="skill_demo/examples",
        help="Root directory containing Skill subdirectories. Defaults to skill_demo/examples.",
    )
    parser.add_argument("--module", default=None, help="Optional current web module context.")
    parser.add_argument("--page", default=None, help="Optional current page or route context.")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Call an OpenAI-compatible LLM using config/llm.local.json or env vars.",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to local LLM JSON config. Defaults to config/llm.local.json.",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = answer_question(
        args.question,
        args.skills_root,
        module=args.module,
        page=args.page,
        llm_client=build_env_llm_client(force=args.use_llm, config_path=args.llm_config),
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

