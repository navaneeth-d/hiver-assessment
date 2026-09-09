"""
End-to-end unit tests for AppleSupportAgent.
"""

from support_agent.agent import AppleSupportAgent
from support_agent.schemas import Intent


def test_agent_end_to_end_battery():
    agent = AppleSupportAgent(use_llm=False)
    query = "My iPhone 7 battery is draining super quickly and gets hot to the touch."
    out = agent.process(query)

    assert out.intent == Intent.BATTERY_POWER_CHARGING
    assert out.intent_confidence > 0.6
    assert out.escalation.escalate is False
    assert "Settings > Battery" in out.reply
    assert out.execution_time_ms > 0
    assert len(out.grounded_resolutions) > 0


def test_agent_end_to_end_safety_hazard():
    agent = AppleSupportAgent(use_llm=False)
    query = "My iPhone battery has swollen up and is bursting out of the casing! Is this a fire risk?"
    out = agent.process(query)

    assert out.escalation.escalate is True
    assert "SAFETY HAZARD" in out.escalation.reason
    assert "stop using" in out.reply.lower()
    assert "do not attempt to charge" in out.reply.lower()


def test_agent_end_to_end_account_theft():
    agent = AppleSupportAgent(use_llm=False)
    query = "Someone changed my Apple ID password from another country! I am locked out!"
    out = agent.process(query)

    assert out.intent == Intent.ACCOUNT_SECURITY_BILLING
    assert out.escalation.escalate is True
    assert "iforgot.apple.com" in out.reply or "DM" in out.reply
