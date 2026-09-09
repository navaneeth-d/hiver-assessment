"""
Unit tests for ResponseGenerator.
"""

import pytest
from support_agent.response_generator import DM_LINK, ResponseGenerator
from support_agent.schemas import (
    EscalationDecision,
    EscalationReasonCategory,
    Intent,
)


@pytest.fixture(scope="module")
def generator():
    return ResponseGenerator(use_llm=False)


def test_response_contains_dm_link(generator):
    decision = EscalationDecision(
        escalate=False,
        category=EscalationReasonCategory.NONE_STANDARD_TROUBLESHOOTING,
        reason="Standard guidance",
        confidence=0.90,
    )
    reply = generator.generate(
        query="My battery drains fast on iOS 11",
        intent=Intent.BATTERY_POWER_CHARGING,
        escalation=decision,
        grounded_resolutions=[],
    )
    assert DM_LINK in reply
    assert "Settings > Battery" in reply


def test_safety_hazard_warning_response(generator):
    decision = EscalationDecision(
        escalate=True,
        category=EscalationReasonCategory.SAFETY_HARDWARE_DAMAGE,
        reason="Swollen battery hazard",
        confidence=1.0,
    )
    reply = generator.generate(
        query="My battery is swollen and popping the screen off",
        intent=Intent.HARDWARE_PHYSICAL_DAMAGE,
        escalation=decision,
        grounded_resolutions=[],
    )
    assert "stop using" in reply.lower()
    assert "do not attempt to charge" in reply.lower()
    assert DM_LINK in reply


def test_hardware_damage_escalation_reply(generator):
    decision = EscalationDecision(
        escalate=True,
        category=EscalationReasonCategory.SAFETY_HARDWARE_DAMAGE,
        reason="Screen cracked",
        confidence=0.98,
    )
    reply = generator.generate(
        query="I dropped my phone and cracked the screen",
        intent=Intent.HARDWARE_PHYSICAL_DAMAGE,
        escalation=decision,
        grounded_resolutions=[],
    )
    assert "apple.com/retail" in reply or "authorized service" in reply.lower()
    assert DM_LINK in reply
