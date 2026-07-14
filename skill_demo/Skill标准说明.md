# Skill 标准说明

## 1. 标准定位

本项目 Skill 采用 LangChain Deep Agents 兼容的 Agent Skills 目录规范。每个 Skill 独立成目录，目录内必须包含 `SKILL.md` 文件；系统启动或刷新时读取各 Skill 的 frontmatter，主要依据 `name` 与 `description` 完成候选路由，命中后再加载完整 `SKILL.md` 内容组织步骤化回答。

Skill 主要用于沉淀标准操作流程、业务专家经验、常见异常处理和验证方式。普通用户通过 Chatbot 提问时，系统根据问题命中对应 Skill，并按 Skill 中的步骤生成回答。

## 2. 目录结构

```text
skills/
└── create-inspection-task/
    ├── SKILL.md
    ├── assets/
    │   ├── step-1.png
    │   └── step-2.png
    ├── references/
    │   └── 相关资料.md
    └── scripts/
        └── validate.py
```

## 3. 标准顶层字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | 是 | 英文短名，建议与目录名一致，仅使用小写字母、数字和连字符 |
| `description` | 是 | 路由核心字段，说明做什么、什么时候用，并写入典型用户问法 |
| `license` | 否 | 许可声明，私有项目写 `proprietary` |
| `compatibility` | 否 | 技术兼容性、适用系统或前置依赖说明 |
| `metadata` | 否 | 业务字段统一放入此字典，避免污染标准顶层字段 |
| `allowed-tools` | 否 | 允许调用的工具白名单；本期操作说明类 Skill 可为空数组 |

## 4. 编写要求

1. 一个 Skill 只描述一类明确操作，不把创建、修改、删除混在一起。
2. `description` 必须写清楚典型问法，例如“怎么建巡检任务”“派单怎么操作”。
3. 每个步骤包含“操作”和“界面位置描述”。界面位置描述只用于 Chatbot 告诉用户去哪个页面、哪个区域操作，不作为前端高亮或锚点字段。
4. 必须包含验证方式，说明用户如何确认操作完成。
5. 建议包含常见异常和处理办法，减少重复答疑。
6. 截图放在 `assets/`，长篇说明放在 `references/`，确定性校验脚本放在 `scripts/`。

## 5. 运行方式

工程侧将 Skill 目录放入运行环境的 `skills/` 目录。LangGraph / Deep Agents 启动或刷新时扫描 Skill frontmatter，建立候选 Skill 列表；当用户问题命中某个 Skill 后，再读取完整 `SKILL.md` 并生成步骤化回答。
