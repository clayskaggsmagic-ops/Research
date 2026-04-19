"""
Shared I/O helpers for the experiment runners.

- Loads the pinned config, question manifest, and prompt templates.
- Deterministic hashing for prompt + briefing reproducibility.
- Append-only JSONL writer for prediction records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # allowed to be absent; only config.load_config needs it


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(rel: str | Path) -> Path:
    """Resolve a path relative to the repo root."""
    return (REPO_ROOT / rel).resolve()


def load_config(path: str | Path = "evaluation_plan/config.yaml") -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed; `pip install pyyaml` or read config manually")
    return yaml.safe_load(repo_path(path).read_text())


def load_manifest(path: str | Path) -> list[dict]:
    data = json.loads(repo_path(path).read_text())
    return data["questions"]


def load_prompt(name: str, prompts_dir: str | Path = "evaluation_plan/prompts") -> str:
    """Return the raw text of a prompt template (e.g. 'trump_system.md')."""
    return repo_path(Path(prompts_dir) / name).read_text()


def sha256_short(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def append_prediction(jsonl_path: str | Path, record: dict) -> None:
    """Append one prediction record as a single JSON line. Creates parent dirs."""
    p = Path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def already_predicted(jsonl_path: str | Path, question_id: str, sample_idx: int) -> bool:
    """Cheap resume check — True if this (qid, sample_idx) row already exists in the file."""
    p = Path(jsonl_path)
    if not p.exists():
        return False
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("question_id") == question_id and row.get("sample_idx") == sample_idx:
            return True
    return False
