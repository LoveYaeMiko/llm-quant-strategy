# FQA

> 项目定位待定 — 当前为初始化脚手架。

## 项目状态

- 本仓库已初始化 git，并配置了 Claude Code 工程技能插件（`feature-dev` 与 `mattpocock-skills`）。
- 项目具体用途尚未确定，此文件后续随项目定位更新。

## 可用的工程技能

本仓库项目级启用了以下 Claude Code 插件（另在用户级也已启用）：

### feature-dev

- `/feature-dev <功能描述>` — 7 阶段特性开发工作流（发现 → 代码库探索 → 澄清问题 → 架构设计 → 实现 → 质量审查 → 总结）。
- 附带 Agents：
  - `code-explorer` — 深入分析现有代码库，追踪执行路径
  - `code-architect` — 基于代码库惯例设计特性架构与实现蓝图
  - `code-reviewer` — 审查 Bug、逻辑错误、安全漏洞与代码质量

### mattpocock-skills

- `diagnosing-bugs` — 疑难 Bug 与性能回归的诊断循环
- `tdd` — 测试驱动开发
- `prototype` — 快速原型
- `research` — 调研
- `domain-modeling` — 领域建模
- `codebase-design` — 代码库架构设计
- `code-review` — 代码审查
- `resolving-merge-conflicts` — 解决合并冲突
- `wizard` — 向导式流程
- `grilling` — 挑战式追问，验证想法
- `writing-for-agents` — 面向 agent 的写作规范

（插件还包含其他由用户显式调用的技能，如 `setup-matt-pocock-skills` 等。）

## Agent skills

### Issue tracker

Issue 以本地 markdown 文件形式存放在 `.scratch/<feature-slug>/`，每个特性一个目录，一个 ticket 一个文件。参见 `docs/agents/issue-tracker.md`。

### Domain docs

单上下文（single-context）布局：仓库根目录一个 `CONTEXT.md` + `docs/adr/`。参见 `docs/agents/domain.md`。
