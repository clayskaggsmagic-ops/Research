"""
Standalone final export — loads checkpoint_stage3.json (122 refined Qs),
applies programmatic quality filters, text-based dedup, difficulty scoring,
and exports the final manifest as markdown.

No LLM API keys required. All filtering is rule-based.

Usage:
    python -m src.export_final
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/tmp/pipeline_output")


def load_checkpoint() -> list[dict]:
    """Load refined questions from Stage 3 checkpoint."""
    ckpt = OUTPUT_DIR / "checkpoint_stage3.json"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")
    data = json.loads(ckpt.read_text())
    return data.get("refined_questions", [])


def quality_filter(questions: list[dict]) -> list[dict]:
    """
    Programmatic quality gate — removes questions with known issues
    based on Stage 2.5 research flags already embedded in the data.
    """
    kept = []
    for q in questions:
        qid = q.get("question_id", "?")
        flags = q.get("research_flags", [])
        quality = q.get("research_quality", "")
        
        # Drop questions flagged as DROP in research
        if quality == "DROP":
            logger.info("DROP (research): %s", qid)
            continue
        
        # Drop if any flag mentions ALREADY_RESOLVED or NOT_LEADER_DECISION
        flag_text = " ".join(flags).lower() if flags else ""
        if "already_resolved" in flag_text:
            logger.info("DROP (already_resolved): %s", qid)
            continue
        if "not_leader_decision" in flag_text:
            logger.info("DROP (not_leader_decision): %s", qid)
            continue
        
        # Require Trump's name in question text
        q_text = q.get("question_text", "").lower()
        if "trump" not in q_text:
            logger.info("DROP (no Trump in question_text): %s", qid)
            continue
        
        # Require question_text to be non-empty
        if not q.get("question_text", "").strip():
            logger.info("DROP (empty text): %s", qid)
            continue
        
        # Require resolution_date
        if not q.get("resolution_date"):
            logger.info("DROP (no resolution_date): %s", qid)
            continue
        
        # Require at least some resolution criteria
        if not q.get("resolution_criteria") and not q.get("resolution_source"):
            logger.info("DROP (no resolution criteria): %s", qid)
            continue
        
        kept.append(q)
    
    logger.info("Quality filter: %d → %d (dropped %d)", len(questions), len(kept), len(questions) - len(kept))
    return kept


def text_dedup(questions: list[dict], threshold: float = 0.75) -> list[dict]:
    """
    Remove near-duplicate questions using SequenceMatcher on question_text.
    Keeps the version with more detailed resolution_criteria.
    """
    kept: list[dict] = []
    
    for q in questions:
        q_text = q.get("question_text", "").lower().strip()
        is_dup = False
        
        for existing in kept:
            e_text = existing.get("question_text", "").lower().strip()
            ratio = SequenceMatcher(None, q_text, e_text).ratio()
            
            if ratio >= threshold:
                # Keep the one with better resolution criteria
                q_crit = len(q.get("resolution_criteria", "") or "")
                e_crit = len(existing.get("resolution_criteria", "") or "")
                if q_crit > e_crit:
                    kept.remove(existing)
                    kept.append(q)
                    logger.info("DEDUP: replaced %s with %s (ratio=%.2f)", 
                                existing.get("question_id"), q.get("question_id"), ratio)
                else:
                    logger.info("DEDUP: dropped %s (dup of %s, ratio=%.2f)", 
                                q.get("question_id"), existing.get("question_id"), ratio)
                is_dup = True
                break
        
        if not is_dup:
            kept.append(q)
    
    logger.info("Dedup: %d → %d (removed %d dupes)", len(questions), len(kept), len(questions) - len(kept))
    return kept


def score_difficulty(q: dict) -> str:
    """
    Heuristic difficulty scoring based on base_rate, question complexity,
    and question type. Aims for ~20/60/20 easy/medium/hard.
    """
    base_rate = q.get("base_rate", 0.5)
    q_type = q.get("question_type", "binary")
    options = q.get("options", [])
    bg_len = len(q.get("background", "") or "")
    
    # Extreme base rates = easier to predict
    rate_distance = abs(base_rate - 0.5)
    
    # MC with 4+ options is inherently harder
    if q_type == "action_selection" and len(options) >= 4:
        if rate_distance > 0.35:
            return "easy"
        elif rate_distance > 0.2 or bg_len > 800:
            return "medium"
        else:
            return "hard"
    # MC with fewer options
    elif q_type == "action_selection":
        if rate_distance > 0.3:
            return "easy"
        elif rate_distance > 0.15:
            return "medium"
        else:
            return "hard"
    else:  # binary
        if rate_distance > 0.35:
            return "easy"
        elif rate_distance > 0.1:
            return "medium"
        else:
            return "hard"


def compute_distribution(questions: list[dict]) -> dict:
    """Compute domain/difficulty/format distribution."""
    domains = Counter(q.get("domain", "unknown") for q in questions)
    difficulties = Counter(q.get("difficulty", "medium") for q in questions)
    formats = Counter(q.get("question_type", "binary") for q in questions)
    return {
        "total": len(questions),
        "domains": dict(domains.most_common()),
        "difficulties": dict(difficulties.most_common()),
        "formats": dict(formats.most_common()),
    }


def export_markdown(questions: list[dict], output_path: Path):
    """Export final question manifest as structured markdown."""
    
    # Score difficulty for each question
    for q in questions:
        q["difficulty"] = score_difficulty(q)
    
    dist = compute_distribution(questions)
    
    lines = [
        "# Trump Decision Forecasting — Final Question Manifest",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Questions:** {dist['total']}",
        f"**Pipeline:** Bosse et al. (2026) adapted — Stages 1-3 complete, programmatic verification",
        f"**Model:** gemini-2.5-flash (all stages)",
        "",
        "## Distribution Summary",
        "",
        "### By Domain",
        "| Domain | Count | % |",
        "|--------|-------|---|",
    ]
    
    for domain, count in dist["domains"].items():
        pct = count / dist["total"] * 100
        lines.append(f"| {domain} | {count} | {pct:.0f}% |")
    
    lines.extend([
        "",
        "### By Difficulty",
        "| Difficulty | Count | % |",
        "|------------|-------|---|",
    ])
    for diff, count in dist["difficulties"].items():
        pct = count / dist["total"] * 100
        lines.append(f"| {diff} | {count} | {pct:.0f}% |")
    
    lines.extend([
        "",
        "### By Format",
        "| Format | Count | % |",
        "|--------|-------|---|",
    ])
    for fmt, count in dist["formats"].items():
        pct = count / dist["total"] * 100
        lines.append(f"| {fmt} | {count} | {pct:.0f}% |")
    
    lines.extend(["", "---", "", "## Questions", ""])
    
    # Group by domain
    by_domain: dict[str, list[dict]] = {}
    for q in questions:
        d = q.get("domain", "unknown")
        by_domain.setdefault(d, []).append(q)
    
    q_num = 0
    for domain in sorted(by_domain.keys()):
        domain_qs = by_domain[domain]
        lines.append(f"### {domain.replace('_', ' ').title()} ({len(domain_qs)} questions)")
        lines.append("")
        
        for q in domain_qs:
            q_num += 1
            qid = q.get("question_id", f"Q-{q_num}")
            q_type = q.get("question_type", "binary")
            difficulty = q.get("difficulty", "medium")
            base_rate = q.get("base_rate", 0.5)
            res_date = q.get("resolution_date", "TBD")
            
            lines.append(f"#### {q_num}. [{qid}] {q.get('question_text', 'N/A')}")
            lines.append("")
            lines.append(f"- **Type:** {q_type} | **Difficulty:** {difficulty} | **Base Rate:** {base_rate:.2f}")
            lines.append(f"- **Domain:** {domain} | **Resolution Date:** {res_date}")
            
            # Options for MC questions
            options = q.get("options", [])
            if options and q_type == "action_selection":
                lines.append("- **Options:**")
                for i, opt in enumerate(options):
                    if isinstance(opt, dict):
                        label = opt.get("label", chr(65 + i))
                        text = opt.get("text", str(opt))
                        lines.append(f"  - **{label}:** {text}")
                    else:
                        lines.append(f"  - **{chr(65 + i)}:** {opt}")
            
            # Background
            bg = q.get("background", "")
            if bg:
                lines.append(f"- **Background:** {bg[:500]}{'...' if len(bg) > 500 else ''}")
            
            # Resolution criteria
            rc = q.get("resolution_criteria", "")
            if rc:
                lines.append(f"- **Resolution Criteria:** {rc[:400]}{'...' if len(rc) > 400 else ''}")
            
            # Resolution source
            rs = q.get("resolution_source", "")
            if rs:
                lines.append(f"- **Resolution Source:** {rs}")
            
            # Fine print
            fp = q.get("fine_print", "")
            if fp:
                lines.append(f"- **Fine Print:** {fp[:300]}{'...' if len(fp) > 300 else ''}")
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    output_path.write_text("\n".join(lines))
    logger.info("Exported %d questions to %s", len(questions), output_path)
    return output_path


def export_json(questions: list[dict], output_path: Path):
    """Export as structured JSON manifest."""
    manifest = {
        "version": "1.0",
        "generated": datetime.now().isoformat(),
        "pipeline": "bosse2026-adapted",
        "model": "gemini-2.5-flash",
        "total_questions": len(questions),
        "distribution": compute_distribution(questions),
        "questions": questions,
    }
    output_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Exported JSON manifest to %s", output_path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Load
    logger.info("Loading Stage 3 checkpoint...")
    questions = load_checkpoint()
    logger.info("Loaded %d refined questions", len(questions))
    
    # Filter
    logger.info("Running quality filter...")
    questions = quality_filter(questions)
    
    # Dedup
    logger.info("Running text dedup...")
    questions = text_dedup(questions)
    
    # Score difficulty
    for q in questions:
        q["difficulty"] = score_difficulty(q)
    
    # Distribution check
    dist = compute_distribution(questions)
    logger.info("Final distribution: %s", json.dumps(dist, indent=2))
    
    # Export
    md_path = OUTPUT_DIR / "final_questions.md"
    json_path = OUTPUT_DIR / "final_manifest.json"
    
    export_markdown(questions, md_path)
    export_json(questions, json_path)
    
    logger.info(
        "\n═══ EXPORT COMPLETE ═══\n"
        "  Input: 122 refined questions\n"
        "  After quality filter: %d\n"
        "  After dedup: %d\n"
        "  Markdown: %s\n"
        "  JSON: %s",
        len(questions), len(questions),
        md_path, json_path,
    )


if __name__ == "__main__":
    main()
