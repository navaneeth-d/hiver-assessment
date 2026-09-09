"""
Unit tests for TrivialBaselineAgent and SimpleBaselineAgent.
"""

from support_agent.baselines import SimpleBaselineAgent, TrivialBaselineAgent
from support_agent.schemas import Intent


def test_trivial_baseline_always_majority():
    agent = TrivialBaselineAgent()
    out1 = agent.process("My battery is dying and phone on fire")
    assert out1.intent == Intent.SOFTWARE_UPDATE_OS
    assert out1.escalation.escalate is False
    assert "Thanks for reaching out" in out1.reply

    out2 = agent.process("Screen cracked")
    assert out2.intent == Intent.SOFTWARE_UPDATE_OS
    assert out2.escalation.escalate is False


def test_simple_baseline_keyword_matching():
    agent = SimpleBaselineAgent()
    out_battery = agent.process("My battery is draining fast")
    assert out_battery.intent == Intent.BATTERY_POWER_CHARGING

    out_screen = agent.process("My screen is broken")
    assert out_screen.intent == Intent.HARDWARE_PHYSICAL_DAMAGE

    # Test keyword escalation
    out_agent = agent.process("I want to speak to an agent")
    assert out_agent.escalation.escalate is True
    assert "agent" in out_agent.escalation.reason

    # Test non-escalated query without keywords
    out_normal = agent.process("My wifi keeps dropping")
    assert out_normal.escalation.escalate is False
