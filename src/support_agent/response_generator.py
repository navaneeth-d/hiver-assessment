"""
Response Generation Module for AppleSupport AI Agent.
Drafts empathetic, brand-consistent replies grounded in historical Apple support
resolutions and official diagnostic protocols. Supports both LLM synthesis and
robust deterministic offline synthesis.
"""

import os
import re
from typing import List, Optional

from .schemas import (
    EscalationDecision,
    Intent,
    RetrievedResolution,
)


DM_LINK = "https://twitter.com/messages/compose?recipient_id=AppleSupport"


class ResponseGenerator:
    """
    Generates grounded customer support replies reflecting Apple's brand voice,
    historical resolution patterns, and escalation directives.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", use_llm: bool = False):
        self.model_name = model_name
        self.use_llm = use_llm
        self._check_api_availability()

    def _check_api_availability(self):
        """Check if any LLM API key is present in environment."""
        self.has_openai = bool(os.getenv("OPENAI_API_KEY"))
        self.has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        self.has_gemini = bool(os.getenv("GEMINI_API_KEY"))

    def generate(
        self,
        query: str,
        intent: Intent,
        escalation: EscalationDecision,
        grounded_resolutions: List[RetrievedResolution],
        protocol: Optional[str] = None,
    ) -> str:
        """
        Draft a response grounded in historical resolutions and escalation decision.
        """
        if self.use_llm and (self.has_openai or self.has_anthropic or self.has_gemini):
            try:
                return self._generate_llm(query, intent, escalation, grounded_resolutions, protocol)
            except Exception as e:
                print(f"LLM generation failed: {e}. Falling back to grounded synthesis.")

        return self._generate_offline(query, intent, escalation, grounded_resolutions, protocol)

    def _generate_offline(
        self,
        query: str,
        intent: Intent,
        escalation: EscalationDecision,
        grounded_resolutions: List[RetrievedResolution],
        protocol: Optional[str] = None,
    ) -> str:
        """
        Deterministic, offline generation grounded in top historical resolution
        and canonical Apple support patterns.
        """
        q_lower = query.lower()

        # 1. Handle Critical Safety Hazard
        if (
            "swollen" in q_lower
            or "bulging" in q_lower
            or "fire" in q_lower
            or "smoke" in q_lower
            or "expand" in q_lower
            or "popping off" in q_lower
            or "burn" in q_lower
        ):
            return (
                f"We want to ensure your safety. Please stop using and do not attempt to charge your device immediately. "
                f"We recommend having it physically inspected right away at an Apple Store or Authorized Service Provider. "
                f"DM us your postal code so we can help schedule an immediate appointment: {DM_LINK}"
            )

        # 2. Handle Escalated Hardware Damage
        if escalation.escalate and (
            intent == Intent.HARDWARE_PHYSICAL_DAMAGE
            or escalation.category.value == "safety_hardware_damage"
        ):
            return (
                f"We understand how concerning hardware issues can be. Physical damage requires hands-on diagnostic testing by a technician. "
                f"You can find your nearest Apple Store or Authorized Service Provider at apple.com/retail, "
                f"or send us a DM so we can explore repair and warranty options together: {DM_LINK}"
            )

        # 3. Handle Escalated Account Security / Billing
        if escalation.escalate and (
            intent == Intent.ACCOUNT_SECURITY_BILLING
            or escalation.category.value == "account_security_billing"
        ):
            if any(w in q_lower for w in ["charge", "refund", "billed", "money", "itunes", "purchase"]):
                return (
                    f"We take billing concerns very seriously and want to help investigate right away. "
                    f"You can review your purchase history anytime at reportaproblem.apple.com. "
                    f"To protect your privacy and credentials, please join us in DM so we can verify the transactions: {DM_LINK}"
                )
            return (
                f"We want to help secure your account. For your safety, never share passwords or verification codes publicly. "
                f"You can start account recovery at iforgot.apple.com, or send us a DM so we can guide you through the process: {DM_LINK}"
            )

        # 4. Handle Customer Distress / Legal
        if escalation.escalate and escalation.category.value == "customer_distress_legal":
            return (
                f"We are very sorry for the frustration this has caused and we want to turn things around. "
                f"Please send us a direct message with your case number and preferred contact information so a senior specialist can review your situation immediately: {DM_LINK}"
            )

        # 5. Handle Order / Delivery Dispute
        if escalation.escalate and (
            intent == Intent.STORE_ORDERS_RESERVATIONS
            or escalation.category.value == "order_logistics_dispute"
        ):
            return (
                f"We know how eager you are for your order and want to make sure it gets to you safely. "
                f"You can check current tracking at apple.com/orderstatus, or send us a DM with your Apple Order number so we can investigate courier dispatch: {DM_LINK}"
            )

        # 6. Handle Ambiguous / Low Confidence
        if escalation.escalate and escalation.category.value == "ambiguous_low_confidence":
            return (
                f"We're here to help! Could you provide a bit more detail on what device you're using and what specifically is happening? "
                f"Feel free to reply with your iOS version or meet us in DM: {DM_LINK}"
            )

        # Fallback for any other escalated cases
        if escalation.escalate:
            return (
                f"We want to make sure you get the right support for this issue. "
                f"Please send us a direct message so a support specialist can look into this with you privately: {DM_LINK}"
            )

        # 7. Auto-Handled Standard Troubleshooting: Grounded in Historical Resolutions
        device_mention = "your device"
        for dev in ["iphone", "ipad", "macbook", "apple watch", "airpods"]:
            if dev in q_lower:
                device_mention = dev.title()
                break

        # Check if top historical resolution exists and contains concrete troubleshooting guidance
        if grounded_resolutions:
            best_res = grounded_resolutions[0]
            ref_clean = re.sub(r"^(@\w+\s*)+", "", best_res.reply).strip()
            # If historical reply is substantive and contains concrete diagnostic steps
            has_diagnostic = any(kw in ref_clean.lower() for kw in ["settings", "restart", "reset", "version", "update", "toggle"])
            if len(ref_clean) > 40 and has_diagnostic:
                if "http" not in ref_clean:
                    ref_clean = f"{ref_clean} DM us here: {DM_LINK}"
                return ref_clean

        # Canonical troubleshooting fallbacks by intent
        if intent == Intent.BATTERY_POWER_CHARGING:
            return (
                f"We'd love to help get the most out of your {device_mention}'s battery. "
                f"Head to Settings > Battery to see which apps are consuming the most power, and check Battery Health. "
                f"Which iOS version are you on (Settings > General > About)? Reply in DM: {DM_LINK}"
            )
        elif intent == Intent.SOFTWARE_UPDATE_OS:
            return (
                f"We're here to help get {device_mention} running smoothly. Have you tried a forced restart since updating? "
                f"Also, let us know your exact iOS version from Settings > General > About via DM: {DM_LINK}"
            )
        elif intent == Intent.CONNECTIVITY_NETWORK:
            return (
                f"We can help with connection issues. Try toggling Airplane mode on for 15 seconds, or Reset Network Settings via Settings > General > Reset. "
                f"Let us know if that helps, or send a DM: {DM_LINK}"
            )
        elif intent == Intent.APPS_MEDIA_FEATURES:
            return (
                f"We can help get your apps working properly. Have you tried force-closing the app and checking the App Store for updates? "
                f"If the issue persists, reach out in DM: {DM_LINK}"
            )
        elif intent == Intent.STORE_ORDERS_RESERVATIONS:
            return (
                f"You can check the latest status of your order anytime at apple.com/orderstatus with your order number. "
                f"Let us know if you need further assistance via DM: {DM_LINK}"
            )
        else:
            return (
                f"We're here to help. Could you let us know what iOS version is installed under Settings > General > About? "
                f"Send us a DM so we can troubleshoot together: {DM_LINK}"
            )

    def _generate_llm(
        self,
        query: str,
        intent: Intent,
        escalation: EscalationDecision,
        grounded_resolutions: List[RetrievedResolution],
        protocol: Optional[str],
    ) -> str:
        """Generate response via OpenAI API with grounding and brand voice constraints."""
        import openai

        hist_context = "\n".join(
            [f"- Historical Customer: {r.customer_query}\n  Apple Reply: {r.reply}" for r in grounded_resolutions[:2]]
        )

        system_prompt = (
            "You are an expert AI support agent for Apple Support on Twitter (@AppleSupport).\n"
            "Brand Guidelines:\n"
            "- Tone: Empathetic, calm, professional, concise (Twitter style, max 280 characters if possible).\n"
            "- Never ask for passwords, credit card info, or PII publicly.\n"
            "- When asking for diagnostics, specify path: Settings > General > About.\n"
            "- If escalated, invite them to private DM or recommend Genius Bar/Authorized Service Provider.\n"
            f"- Always include the official DM link: {DM_LINK}\n"
        )

        user_prompt = (
            f"Customer Message: \"{query}\"\n"
            f"Classified Intent: {intent.value}\n"
            f"Escalation Decision: {'ESCALATE TO HUMAN' if escalation.escalate else 'AUTO-HANDLE'}\n"
            f"Escalation Reason: {escalation.reason}\n"
            f"Troubleshooting Protocol: {protocol or 'Standard'}\n\n"
            f"Historical Similar Resolutions:\n{hist_context}\n\n"
            "Draft the final Apple Support tweet reply now."
        )

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
