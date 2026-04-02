"""
Tests for Stage 2 — Proto-Question Generator.

Schema validation tests are pure unit tests (no API needed).
Agent tests hit LIVE APIs — requires GOOGLE_API_KEY and TAVILY_API_KEY in .env.
"""

from __future__ import annotations

import pytest

from src.schemas import (
    DecisionSeed,
    DomainType,
    PipelineConfig,
    PipelineState,
    Question,
    QuestionType,
    Source,
)
from src.stages.stage2_proto_questions import (
    GeneratedQuestion,
    ProtoQuestionResponse,
    run_seed_agent,
    run_stage2,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_seed(seed_id: str, domain: DomainType) -> DecisionSeed:
    """Create a test seed with realistic data."""
    return DecisionSeed(
        seed_id=seed_id,
        event_description=(
            "President Trump announced a new round of tariffs on Chinese imports, "
            "raising rates from 10% to 25% on $200 billion worth of goods. The move "
            "escalated the US-China trade war amid stalled negotiations."
        ),
        decision_taken="Imposed 25% tariffs on $200B of Chinese imports",
        decision_date="2025-06-15",
        simulation_date="2025-06-01",
        domain=domain,
        plausible_alternatives=[
            "Maintain tariffs at 10%",
            "Raise tariffs to 50%",
            "Remove tariffs entirely as a goodwill gesture",
            "Impose targeted tariffs on specific sectors only",
            "Take no action",
        ],
        sources=[Source(name="Reuters", url="https://reuters.com/trade", date="2025-06-15")],
        attribution_evidence="President personally announced the tariff increase via Truth Social",
        leader_attributable=True,
    )


# ── Schema Tests (no API needed) ──────────────────────────────────────────────


class TestSchemas:
    """Pure unit tests for Pydantic schemas — no API calls."""

    def test_binary_question_schema(self):
        q = GeneratedQuestion(
            title="Test binary",
            background="Context here",
            question_text="Will X happen by Y?",
            question_type="binary",
            options=None,
            rationale="Good question because...",
        )
        assert q.question_type == "binary"
        assert q.options is None

    def test_action_selection_schema(self):
        q = GeneratedQuestion(
            title="Test action",
            background="Context here",
            question_text="Which action?",
            question_type="action_selection",
            options=["A", "B", "C", "Take no action"],
            rationale="Diverse options.",
        )
        assert q.question_type == "action_selection"
        assert len(q.options) == 4

    def test_proto_response_schema(self):
        resp = ProtoQuestionResponse(
            seed_id="SEED-001",
            questions=[
                GeneratedQuestion(
                    title="Q1",
                    background="Bg",
                    question_text="Question?",
                    question_type="binary",
                    options=None,
                    rationale="Reason",
                )
            ],
            research_summary="Researched stuff.",
        )
        assert resp.seed_id == "SEED-001"
        assert len(resp.questions) == 1


# ── Live Agent Tests (require API keys) ───────────────────────────────────────


class TestLiveAgent:
    """Integration tests against live Gemini + Tavily APIs."""

    @pytest.mark.asyncio
    async def test_run_seed_agent_produces_questions(self):
        """Run a single seed through the live agent and verify output."""
        seed = _make_seed("SEED-LIVE-001", DomainType.TRADE_TARIFFS)

        questions = await run_seed_agent(
            seed=seed,
            model_name="gemini-3-flash",
            temperature=0.7,
        )

        # Should produce at least 1 question
        assert len(questions) >= 1, f"Expected ≥1 questions, got {len(questions)}"

        for q in questions:
            # Must be valid Question objects
            assert isinstance(q, Question)

            # Core fields filled
            assert q.question_id.startswith("Q-SEED-LIVE-001-")
            assert q.seed_id == "SEED-LIVE-001"
            assert q.domain == DomainType.TRADE_TARIFFS
            assert q.simulation_date == "2025-06-01"
            assert q.title
            assert q.background
            assert q.question_text

            # Question type must be valid
            assert q.question_type in (QuestionType.BINARY, QuestionType.ACTION_SELECTION)

            # If action_selection, must have options with "no action"
            if q.question_type == QuestionType.ACTION_SELECTION:
                assert q.options is not None
                assert len(q.options) >= 3
                assert any("no action" in o.lower() for o in q.options)

            # Resolution fields must be null (proto-question only)
            assert q.correct_answer is None
            assert q.resolution_criteria is None
            assert q.resolution_source is None
            assert q.verification_verdict is None

    @pytest.mark.asyncio
    async def test_run_stage2_end_to_end(self):
        """Run the full Stage 2 orchestration with 2 real seeds."""
        seeds = [
            _make_seed("SEED-E2E-001", DomainType.TRADE_TARIFFS),
            DecisionSeed(
                seed_id="SEED-E2E-002",
                event_description=(
                    "President Trump signed an executive order directing the "
                    "Department of Justice to investigate social media companies "
                    "for alleged political bias and censorship."
                ),
                decision_taken="Signed EO on social media censorship investigation",
                decision_date="2025-05-20",
                simulation_date="2025-05-05",
                domain=DomainType.EXECUTIVE_ORDERS,
                plausible_alternatives=[
                    "Sign the executive order",
                    "Issue a presidential memorandum instead",
                    "Direct FTC investigation without an EO",
                    "Take no action",
                ],
                sources=[Source(name="AP", url="https://apnews.com/eo", date="2025-05-20")],
                attribution_evidence="Signed by the President in the Oval Office",
                leader_attributable=True,
            ),
        ]

        state = PipelineState(
            config=PipelineConfig(
                training_cutoff_date="2025-01-20",
                today_date="2026-04-01",
            ),
            seeds=seeds,
        )

        result = await run_stage2(state)

        # Should produce questions from both seeds
        assert len(result.proto_questions) >= 2, (
            f"Expected ≥2 questions from 2 seeds, got {len(result.proto_questions)}"
        )

        # Both seeds should have questions
        seed_ids = {q.seed_id for q in result.proto_questions}
        assert "SEED-E2E-001" in seed_ids, "Missing questions for SEED-E2E-001"
        assert "SEED-E2E-002" in seed_ids, "Missing questions for SEED-E2E-002"

        # All questions valid
        for q in result.proto_questions:
            assert isinstance(q, Question)
            assert q.title
            assert q.question_text
            assert q.correct_answer is None  # proto only

        # Print results for manual inspection
        print(f"\n{'='*60}")
        print(f"Stage 2 produced {len(result.proto_questions)} proto-questions:")
        print(f"{'='*60}")
        for q in result.proto_questions:
            print(f"\n[{q.question_id}] ({q.question_type.value})")
            print(f"  Title: {q.title}")
            print(f"  Question: {q.question_text}")
            if q.options:
                for j, opt in enumerate(q.options):
                    print(f"    {chr(65+j)}. {opt}")
            print(f"  Simulation date: {q.simulation_date}")
            print(f"  Domain: {q.domain.value}")
