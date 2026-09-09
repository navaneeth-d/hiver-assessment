"""
Unit tests for IntentClassifier.
"""

import pytest
from support_agent.intent_classifier import IntentClassifier
from support_agent.schemas import Intent


@pytest.fixture(scope="module")
def classifier():
    return IntentClassifier()


def test_classify_battery(classifier):
    res = classifier.predict("My iPhone 7 battery percentage drops from 80% to 10% in half an hour")
    assert res.intent == Intent.BATTERY_POWER_CHARGING
    assert res.confidence > 0.5
    assert "battery_power_charging" in res.probabilities


def test_classify_hardware_damage(classifier):
    res = classifier.predict("Dropped my phone and the screen completely shattered into pieces")
    assert res.intent == Intent.HARDWARE_PHYSICAL_DAMAGE
    assert res.confidence > 0.5


def test_classify_software_update(classifier):
    res = classifier.predict("Updated my phone to iOS 11 and now everything is extremely laggy and freezing")
    assert res.intent == Intent.SOFTWARE_UPDATE_OS
    assert res.confidence > 0.5


def test_classify_connectivity(classifier):
    res = classifier.predict("My Wi-Fi keeps disconnecting and my AirPods won't connect via Bluetooth")
    assert res.intent == Intent.CONNECTIVITY_NETWORK
    assert res.confidence > 0.5


def test_classify_account_billing(classifier):
    res = classifier.predict("My Apple ID was locked and I need a refund on an accidental iTunes purchase")
    assert res.intent == Intent.ACCOUNT_SECURITY_BILLING
    assert res.confidence > 0.5


def test_classify_safety_hazard_override(classifier):
    # Tests that swollen battery is classified as hardware damage with high confidence
    res = classifier.predict("My iPhone battery is swollen and bulging out of the case!")
    assert res.intent == Intent.HARDWARE_PHYSICAL_DAMAGE
    assert res.confidence >= 0.95


def test_ambiguity_flag(classifier):
    # Extremely short or vague query should flag ambiguity
    res = classifier.predict("phone thing")
    assert isinstance(res.is_ambiguous, bool)
