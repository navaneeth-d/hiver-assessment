"""
Unit tests for Pydantic schemas and data models.
"""

import pytest
from support_agent.schemas import (
    AgentOutput,
    EscalationDecision,
    EscalationReasonCategory,
    GoldenExample,
    Intent,
    IntentClassificationResult,
    RetrievedResolution,
    RubricScores,
)


def test_intent_enum():
    assert Intent.BATTERY_POWER_CHARGING.value == "battery_power_charging"
    assert Intent.SOFTWARE_UPDATE_OS.value == "software_update_os"
    assert len(Intent) == 7


def test_escalation_decision_model():
    decision = EscalationDecision(
        escalate=True,
        category=EscalationReasonCategory.SAFETY_HARDWARE_DAMAGE,
        reason="Swollen battery hazard",
        confidence=0.99,
    )
    assert decision.escalate is True
    assert decision.category == EscalationReasonCategory.SAFETY_HARDWARE_DAMAGE
    assert "battery" in decision.reason


def test_rubric_scores_validation():
    scores = RubricScores(
        grounding=4.5,
        actionability=4.0,
        tone=5.0,
        escalation=5.0,
        safety=5.0,
        overall=4.6,
    )
    assert scores.overall == 4.6

    with pytest.raises(Exception):
        # Value above 5.0 should trigger validation error
        RubricScores(
            grounding=6.0,
            actionability=4.0,
            tone=5.0,
            escalation=5.0,
            safety=5.0,
            overall=5.2,
        )


def test_agent_output_schema():
    decision = EscalationDecision(
        escalate=False,
        category=EscalationReasonCategory.NONE_STANDARD_TROUBLESHOOTING,
        reason="Standard guidance",
        confidence=0.90,
    )
    res = RetrievedResolution(
        id="123",
        customer_query="battery dying",
        reply="check settings",
        intent="battery_power_charging",
        similarity_score=0.85,
    )
    output = AgentOutput(
        customer_query="battery dying fast",
        intent=Intent.BATTERY_POWER_CHARGING,
        intent_confidence=0.95,
        escalation=decision,
        reply="Please check Settings > Battery.",
        grounded_resolutions=[res],
        suggested_protocol="Battery protocol",
        execution_time_ms=12.5,
        model_mode="offline",
    )
    assert output.intent == Intent.BATTERY_POWER_CHARGING
    assert len(output.grounded_resolutions) == 1
    assert output.execution_time_ms == 12.5
