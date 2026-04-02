---
description: Always use the newest versions of all libraries, models, and APIs
---

# Version Policy: Always Use Latest

**This project mandates the newest stable versions of everything.** My training data may be stale — always verify via web search before choosing a version.

## Models (as of April 2026)

| Role | Model ID | Notes |
|------|----------|-------|
| Fast/cheap (tool-calling, discovery) | `gemini-3-flash` | NOT gemini-2.x |
| Pro reasoning (drafting, refinement) | `gemini-3.1-pro-preview` | Replaced `gemini-3-pro-preview` (discontinued Mar 2026) |
| Cross-model verification | `claude-sonnet-4-20250514` | Different family for adversarial checks |
| Lightweight/bulk | `gemini-3.1-flash-lite-preview` | High-volume, cost-efficient |
| Deep reasoning | `gemini-3.1-deep-think` | Science/research tasks |

## Libraries

| Package | Rule |
|---------|------|
| `langchain-google-genai` | Must be `>=4.0.0` (uses consolidated `google-genai` SDK) |
| `langchain-tavily` | Use this, NOT deprecated `langchain_community.tools.tavily_search` |
| `langgraph` | Use `>=0.4` |
| All others | Run `pip install -U <package>` and check PyPI for latest |

## Before Every Implementation

// turbo-all
1. If choosing a model or library version, **search the web** for the latest docs first
2. Never assume your training data has the correct version — verify
3. When in doubt, check: `pip index versions <package>` or PyPI
