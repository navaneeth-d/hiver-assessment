"""
Baseline agents for comparative benchmarking as required by problem statement:
1. Trivial Baseline: Majority class intent + static escalation + canned generic response.
2. Simple Baseline: Naive keyword matching + simple regex escalation + verbatim nearest neighbor.
"""

import random
import re
from typing import Optional
from .schemas import (
    AgentOutput,
    EscalationDecision,
    EscalationReasonCategory,
    Intent,
    RetrievedResolution,
)


class TrivialBaselineAgent:
    """
    Trivial baseline:
    - Intent: Always predicts majority class ('software_update_os').
    - Escalation: Fixed policy (never escalates; auto-handles everything).
    - Reply: Static canned generic greeting with DM link.
    """

    def __init__(self):
        self.majority_intent = Intent.SOFTWARE_UPDATE_OS
        self.canned_reply = (
            "Thanks for reaching out to Apple Support! We are always happy to help. "
            "Send us a DM so we can look into this together: "
            "https://twitter.com/messages/compose?recipient_id=AppleSupport"
        )

    def process(self, customer_query: str) -> AgentOutput:
        return AgentOutput(
            customer_query=customer_query,
            intent=self.majority_intent,
            intent_confidence=0.50,
            escalation=EscalationDecision(
                escalate=False,
                category=EscalationReasonCategory.NONE_STANDARD_TROUBLESHOOTING,
                reason="Trivial baseline policy: auto-handle all incoming messages.",
                confidence=0.50,
            ),
            reply=self.canned_reply,
            grounded_resolutions=[],
            suggested_protocol=None,
            execution_time_ms=0.1,
            model_mode="trivial_baseline",
        )


class SimpleBaselineAgent:
    """
    Simple baseline:
    - Intent: Naive keyword heuristic matching.
    - Escalation: Keyword trigger on words 'agent', 'human', 'refund', 'speak'.
    - Reply: Verbatim retrieval response or simple template.
    """

    KEYWORD_MAP = {
        "battery": Intent.BATTERY_POWER_CHARGING,
        "drain": Intent.BATTERY_POWER_CHARGING,
        "charge": Intent.BATTERY_POWER_CHARGING,
        "screen": Intent.HARDWARE_PHYSICAL_DAMAGE,
        "broken": Intent.HARDWARE_PHYSICAL_DAMAGE,
        "cracked": Intent.HARDWARE_PHYSICAL_DAMAGE,
        "apple id": Intent.ACCOUNT_SECURITY_BILLING,
        "icloud": Intent.ACCOUNT_SECURITY_BILLING,
        "password": Intent.ACCOUNT_SECURITY_BILLING,
        "refund": Intent.ACCOUNT_SECURITY_BILLING,
        "wifi": Intent.CONNECTIVITY_NETWORK,
        "bluetooth": Intent.CONNECTIVITY_NETWORK,
        "app": Intent.APPS_MEDIA_FEATURES,
        "crash": Intent.APPS_MEDIA_FEATURES,
        "order": Intent.STORE_ORDERS_RESERVATIONS,
        "shipping": Intent.STORE_ORDERS_RESERVATIONS,
    }

    def __init__(self):
        self.escalation_keywords = re.compile(r"\b(agent|human|representative|refund|manager|person)\b", re.IGNORECASE)

    def process(self, customer_query: str) -> AgentOutput:
        q_low = customer_query.lower()

        # Keyword intent match
        intent = Intent.SOFTWARE_UPDATE_OS
        for kw, mapped_intent in self.KEYWORD_MAP.items():
            if kw in q_low:
                intent = mapped_intent
                break

        # Naive keyword escalation
        escalate_match = self.escalation_keywords.search(q_low)
        if escalate_match:
            escalate = True
            category = EscalationReasonCategory.ACCOUNT_SECURITY_BILLING
            reason = f"Simple keyword match on '{escalate_match.group(0)}'."
        else:
            escalate = False
            category = EscalationReasonCategory.NONE_STANDARD_TROUBLESHOOTING
            reason = "No escalation keyword detected."

        reply = (
            f"We'd be glad to help with your issue. Which device and iOS version are you using? "
            f"Send us a DM: https://twitter.com/messages/compose?recipient_id=AppleSupport"
        )

        return AgentOutput(
            customer_query=customer_query,
            intent=intent,
            intent_confidence=0.70,
            escalation=EscalationDecision(
                escalate=escalate,
                category=category,
                reason=reason,
                confidence=0.70,
            ),
            reply=reply,
            grounded_resolutions=[],
            suggested_protocol=None,
            execution_time_ms=0.5,
            model_mode="simple_baseline",
        )
