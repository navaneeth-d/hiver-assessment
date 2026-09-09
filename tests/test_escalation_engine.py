"""
Unit tests for EscalationPolicyEngine.
"""

import pytest
from support_agent.escalation_engine import EscalationPolicyEngine
from support_agent.schemas import (
    EscalationReasonCategory,
    Intent,
    IntentClassificationResult,
)


@pytest.fixture(scope="module")
def engine():
    return EscalationPolicyEngine()


def make_classification(intent: Intent, conf: float = 0.95, ambiguous: bool = False):
    return IntentClassificationResult(
        intent=intent,
        confidence=conf,
        probabilities={intent.value: conf},
        is_ambiguous=ambiguous,
    )


def test_escalate_swollen_battery(engine):
    query = "My battery is swollen and bulging! Is it dangerous?"
    clf = make_classification(Intent.BATTERY_POWER_CHARGING)
    decision = engine.evaluate(query, clf)
    assert decision.escalate is True
    assert decision.category == EscalationReasonCategory.SAFETY_HARDWARE_DAMAGE
    assert "SAFETY HAZARD" in decision.reason


def test_escalate_shattered_screen(engine):
    query = "Dropped my phone on concrete and the glass screen completely cracked"
    clf = make_classification(Intent.HARDWARE_PHYSICAL_DAMAGE)
    decision = engine.evaluate(query, clf)
    assert decision.escalate is True
    assert decision.category == EscalationReasonCategory.SAFETY_HARDWARE_DAMAGE


def test_escalate_account_takeover(engine):
    query = "Someone hacked my Apple ID and changed my recovery email! I am locked out!"
    clf = make_classification(Intent.ACCOUNT_SECURITY_BILLING)
    decision = engine.evaluate(query, clf)
    assert decision.escalate is True
    assert decision.category == EscalationReasonCategory.ACCOUNT_SECURITY_BILLING


def test_escalate_unauthorized_charge(engine):
    query = "There is an unauthorized charge of $150 on my credit card from iTunes and I need a refund!"
    clf = make_classification(Intent.ACCOUNT_SECURITY_BILLING)
    decision = engine.evaluate(query, clf)
    assert decision.escalate is True
    assert decision.category == EscalationReasonCategory.ACCOUNT_SECURITY_BILLING


def test_escalate_legal_threat(engine):
    query = "I have called 5 times and was hung up on. I am contacting my attorney and suing Apple!"
    clf = make_classification(Intent.SOFTWARE_UPDATE_OS)
    decision = engine.evaluate(query, clf)
    assert decision.escalate is True
    assert decision.category == EscalationReasonCategory.CUSTOMER_DISTRESS_LEGAL


def test_escalate_stolen_package(engine):
    query = "UPS said my iPhone was delivered at my door but package was stolen by porch thieves"
    clf = make_classification(Intent.STORE_ORDERS_RESERVATIONS)
    decision = engine.evaluate(query, clf)
    assert decision.escalate is True
    assert decision.category == EscalationReasonCategory.ORDER_LOGISTICS_DISPUTE


def test_auto_handle_standard_software_update(engine):
    query = "Ever since updating to iOS 11 my phone feels a bit slow and laggy"
    clf = make_classification(Intent.SOFTWARE_UPDATE_OS)
    decision = engine.evaluate(query, clf)
    assert decision.escalate is False
    assert decision.category == EscalationReasonCategory.NONE_STANDARD_TROUBLESHOOTING


def test_auto_handle_normal_battery_drain(engine):
    query = "My battery drains pretty fast when using Instagram. Can I charge my iPhone X overnight?"
    clf = make_classification(Intent.BATTERY_POWER_CHARGING)
    decision = engine.evaluate(query, clf)
    assert decision.escalate is False
    assert decision.category == EscalationReasonCategory.NONE_STANDARD_TROUBLESHOOTING


def test_escalate_ambiguous_short_query(engine):
    query = "broken help"
    clf = make_classification(Intent.SOFTWARE_UPDATE_OS, conf=0.20, ambiguous=True)
    decision = engine.evaluate(query, clf)
    assert decision.escalate is True
    assert decision.category == EscalationReasonCategory.AMBIGUOUS_LOW_CONFIDENCE
