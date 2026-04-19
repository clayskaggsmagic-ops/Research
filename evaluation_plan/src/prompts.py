"""
Prompt rendering for the experiment runners.

- Builds the user-message text from a manifest question + optional briefing.
- Normalizes options into A/B/C/... letters for action_selection questions.
- Returns a deterministic prompt hash.
"""

from __future__ import annotations

import string

from evaluation_plan.src.io_utils import load_prompt, sha256_short


OPTION_LETTERS = list(string.ascii_uppercase)  # A..Z


def option_letters_for(question: dict) -> list[str]:
    n = len(question.get("options") or [])
    return OPTION_LETTERS[:n]


def render_options_block(question: dict) -> str:
    opts = question.get("options") or []
    letters = OPTION_LETTERS[: len(opts)]
    return "\n".join(f"{L}. {opt}" for L, opt in zip(letters, opts))


def render_briefing_section(briefing: str | None) -> str:
    if not briefing:
        return ""
    return f"INTELLIGENCE BRIEFING:\n{briefing.strip()}\n"


def render_user_message(question: dict, briefing: str | None) -> str:
    """Fill the binary or action_selection user template for a question."""
    fmt = question["question_type"]
    template_name = "user_binary.md" if fmt == "binary" else "user_action.md"
    tpl = load_prompt(template_name)

    substitutions = {
        "simulation_date": question["simulation_date"],
        "resolution_date": question["resolution_date"],
        "briefing_section": render_briefing_section(briefing),
        "question_id": question["question_id"],
        "question_text": question["question_text"],
        "background": question.get("background", "") or "",
        "resolution_criteria": question.get("resolution_criteria", "") or "",
        "options_block": render_options_block(question) if fmt == "action_selection" else "",
    }
    return tpl.format(**substitutions)


def render_messages(
    question: dict,
    system_prompt_name: str,
    briefing: str | None,
) -> tuple[str, str, str]:
    """
    Build (system_text, user_text, prompt_hash).
    prompt_hash covers system + user text so re-runs hit the same hash only for
    identical prompts.
    """
    system_text = load_prompt(system_prompt_name)
    user_text = render_user_message(question, briefing)
    prompt_hash = sha256_short(system_text + "\n\n" + user_text)
    return system_text, user_text, prompt_hash
