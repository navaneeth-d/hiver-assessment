"""
Escalation Policy Engine for AppleSupport AI Agent.
Evaluates incoming queries against safety, security, customer sentiment,
and ambiguity policies to make an audited auto-handle vs. human escalation decision.
"""

import re
from typing import Optional

from .schemas import (
    EscalationDecision,
    EscalationReasonCategory,
    Intent,
    IntentClassificationResult,
)


class EscalationPolicyEngine:
    """
    Decides whether a customer query should be auto-handled or escalated
    to a human support specialist, providing a concrete stated reason.
    """

    # Physical safety and hardware regex patterns
    PAT_SAFETY_HAZARD = re.compile(
        r"\b(swollen|swelling|bulging|expand\w*|burst|popping off|pop\w* off|smoke|smoking|spark|burn\w*|thermal|cut my finger|sharp glass|fire|hazard)\b",
        re.IGNORECASE,
    )
    PAT_PHYSICAL_DAMAGE = re.compile(
        r"\b(cracked|shattered|broken screen|dropped.*(water|pool|toilet|floor|ground)|water damage|liquid damage|submerged|speaker blown|speaker.*buzz|mic dead|bent|unresponsive touch|lines on screen|flicker|display.*off)\b",
        re.IGNORECASE,
    )

    # Account security and financial disputes
    PAT_ACCOUNT_TAKEOVER = re.compile(
        r"\b(hacked|unauthorized|stolen apple id|compromised|someone logged in|changed my password|russian|phishing|takeover)\b",
        re.IGNORECASE,
    )
    PAT_ACCOUNT_LOCKOUT = re.compile(
        r"\b(locked|locked out|lock out|account disabled|disabled account|forgot password|two factor|2fa|verification code|trusted number|trusted phone|lost phone|lost access)\b",
        re.IGNORECASE,
    )
    PAT_BILLING_DISPUTE = re.compile(
        r"\b(refund|charged me|charged \$\d+|charged a bunch|unauthorized charge|billed twice|double charged|chargeback|fraud|money back|cancel subscription|verify.*billing|itunes bill)\b",
        re.IGNORECASE,
    )

    # Customer distress, frustration, and legal threats
    PAT_LEGAL_THREAT = re.compile(
        r"\b(sue|lawyer|attorney|ftc|bbb|better business bureau|consumer protection|legal action|court|police)\b",
        re.IGNORECASE,
    )
    PAT_HIGH_FRUSTRATION = re.compile(
        r"\b(fuck|shit|asshole|bullshit|scam|thieves|worst company|called \d+ times|agent hung up|disconnect|unacceptable|useless support)\b",
        re.IGNORECASE,
    )

    # Logistics and delivery loss
    PAT_ORDER_LOSS = re.compile(
        r"\b(stolen package|missing delivery|never arrived|porch pirate|package thieves|wrong item delivered|cancelled preorder)\b",
        re.IGNORECASE,
    )

    def evaluate(
        self,
        query: str,
        classification: IntentClassificationResult,
    ) -> EscalationDecision:
        """
        Evaluate customer message and return audited EscalationDecision.
        """
        text = query.strip()
        text_lower = text.lower()

        # 1. Critical Safety Hazard (Swelling, Thermal, Fire)
        if self.PAT_SAFETY_HAZARD.search(text_lower):
            return EscalationDecision(
                escalate=True,
                category=EscalationReasonCategory.SAFETY_HARDWARE_DAMAGE,
                reason=(
                    "CRITICAL SAFETY HAZARD: Physical battery expansion or thermal risk detected. "
                    "Automated handling is prohibited; immediate in-person hardware isolation and Genius Bar referral required."
                ),
                confidence=1.0,
            )

        # 2. Physical Hardware Damage
        if (
            classification.intent == Intent.HARDWARE_PHYSICAL_DAMAGE
            or self.PAT_PHYSICAL_DAMAGE.search(text_lower)
        ):
            return EscalationDecision(
                escalate=True,
                category=EscalationReasonCategory.SAFETY_HARDWARE_DAMAGE,
                reason=(
                    "Physical hardware damage (screen, chassis, or liquid ingress) requires "
                    "hands-on diagnostic testing and physical component repair at an Apple Store or Authorized Service Provider."
                ),
                confidence=0.98,
            )

        # 3. Account Takeover / Active Security Compromise
        if self.PAT_ACCOUNT_TAKEOVER.search(text_lower):
            return EscalationDecision(
                escalate=True,
                category=EscalationReasonCategory.ACCOUNT_SECURITY_BILLING,
                reason=(
                    "Active account security compromise suspected. Public automated triage prohibited; "
                    "requires urgent identity freeze and verification by Apple Security personnel."
                ),
                confidence=0.99,
            )

        # 4. Financial Billing Dispute & Unauthorized Charges (Disambiguated from electrical battery charging)
        financial_billing_terms = [
            "refund", "charged me", "charged my", "charge on my", "unauthorized charge",
            "billed twice", "double charged", "chargeback", "fraud", "money back",
            "cancel subscription", "verify billing", "verify my billing", "itunes bill",
            "charged a bunch", "card", "bank", "statement", "credit card", "debit card"
        ]
        is_billing_kw = any(term in text_lower for term in financial_billing_terms)
        if self.PAT_BILLING_DISPUTE.search(text_lower) or (classification.intent == Intent.ACCOUNT_SECURITY_BILLING and is_billing_kw):
            return EscalationDecision(
                escalate=True,
                category=EscalationReasonCategory.ACCOUNT_SECURITY_BILLING,
                reason=(
                    "Financial transaction or refund dispute involves payment credentials. "
                    "Automated transactions are restricted; requires authorized billing personnel review."
                ),
                confidence=0.95,
            )

        # 5. Account Lockout & 2FA Recovery
        account_lockout_terms = [
            "locked", "disabled", "password", "2fa", "two factor", "verification code",
            "trusted phone", "trusted number", "cannot log in", "can't log in",
            "login failed", "account recovery", "iforgot", "stolen", "lost access to my account"
        ]
        is_lockout_kw = any(term in text_lower for term in account_lockout_terms)
        if self.PAT_ACCOUNT_LOCKOUT.search(text_lower) or (classification.intent == Intent.ACCOUNT_SECURITY_BILLING and is_lockout_kw):
            return EscalationDecision(
                escalate=True,
                category=EscalationReasonCategory.ACCOUNT_SECURITY_BILLING,
                reason=(
                    "Apple ID security lockout or credential recovery requires authenticated, private channel identity verification."
                ),
                confidence=0.92,
            )

        # 6. Logistics and Stolen Delivery Disputes (moved before distress check)
        if self.PAT_ORDER_LOSS.search(text_lower) or (
            classification.intent == Intent.STORE_ORDERS_RESERVATIONS
            and any(w in text_lower for w in ["stolen", "missing", "never arrived", "lost"])
        ):
            return EscalationDecision(
                escalate=True,
                category=EscalationReasonCategory.ORDER_LOGISTICS_DISPUTE,
                reason=(
                    "Lost or stolen delivery requires carrier claims investigation and replacement authorization by human logistics personnel."
                ),
                confidence=0.93,
            )

        # 7. Customer Distress & Legal Threat
        if self.PAT_LEGAL_THREAT.search(text_lower) or self.PAT_HIGH_FRUSTRATION.search(text_lower):
            return EscalationDecision(
                escalate=True,
                category=EscalationReasonCategory.CUSTOMER_DISTRESS_LEGAL,
                reason=(
                    "Customer exhibits severe emotional distress, repeated support failure, or legal/regulatory escalation. "
                    "Demands immediate empathetic intervention by a human senior specialist."
                ),
                confidence=0.96,
            )

        # 8. Ambiguity & Extremely Low Confidence
        words = text.split()
        if len(words) < 4 or classification.is_ambiguous or classification.confidence < 0.28:
            return EscalationDecision(
                escalate=True,
                category=EscalationReasonCategory.AMBIGUOUS_LOW_CONFIDENCE,
                reason=(
                    f"Query lacks sufficient diagnostic details (word count={len(words)}, "
                    f"confidence={classification.confidence:.2f}). Escalating to avoid erroneous automated instructions."
                ),
                confidence=0.85,
            )

        # 9. Standard Self-Serve Auto-Handle
        return EscalationDecision(
            escalate=False,
            category=EscalationReasonCategory.NONE_STANDARD_TROUBLESHOOTING,
            reason=(
                "Issue matches standard diagnostic troubleshooting protocol. "
                "Can be safely and immediately resolved via automated step-by-step guidance."
            ),
            confidence=round(classification.confidence, 4),
        )
