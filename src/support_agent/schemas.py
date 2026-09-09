"""
Data models and schemas for the AppleSupport AI Agent system.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Intent(str, Enum):
    SOFTWARE_UPDATE_OS = "software_update_os"
    BATTERY_POWER_CHARGING = "battery_power_charging"
    HARDWARE_PHYSICAL_DAMAGE = "hardware_physical_damage"
    ACCOUNT_SECURITY_BILLING = "account_security_billing"
    CONNECTIVITY_NETWORK = "connectivity_network"
    APPS_MEDIA_FEATURES = "apps_media_features"
    STORE_ORDERS_RESERVATIONS = "store_orders_reservations"


class EscalationReasonCategory(str, Enum):
    SAFETY_HARDWARE_DAMAGE = "safety_hardware_damage"
    ACCOUNT_SECURITY_BILLING = "account_security_billing"
    CUSTOMER_DISTRESS_LEGAL = "customer_distress_legal"
    ORDER_LOGISTICS_DISPUTE = "order_logistics_dispute"
    AMBIGUOUS_LOW_CONFIDENCE = "ambiguous_low_confidence"
    NONE_STANDARD_TROUBLESHOOTING = "none_standard_troubleshooting"


class EscalationDecision(BaseModel):
    escalate: bool = Field(..., description="Whether to escalate to a human agent (True) or auto-handle (False)")
    category: EscalationReasonCategory = Field(..., description="Categorical classification of the escalation trigger")
    reason: str = Field(..., description="Explicit, audited explanation of why the message was escalated or auto-handled")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the escalation decision")


class RetrievedResolution(BaseModel):
    id: str
    customer_query: str
    reply: str
    intent: str
    similarity_score: float


class IntentClassificationResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    is_ambiguous: bool = False


class AgentOutput(BaseModel):
    customer_query: str
    intent: Intent
    intent_confidence: float
    escalation: EscalationDecision
    reply: str
    grounded_resolutions: List[RetrievedResolution] = Field(default_factory=list)
    suggested_protocol: Optional[str] = None
    execution_time_ms: float = 0.0
    model_mode: str = "offline"


class GoldenExample(BaseModel):
    id: str
    customer_tweet: str
    intent: str
    escalate: bool
    escalation_reason_category: str
    escalation_reason: str
    reference_reply: str
    device_context: Optional[str] = None
    difficulty: str = "medium"
    sampling_stratum: Optional[str] = None
    human_rubric: Optional[Dict[str, float]] = None


class RubricScores(BaseModel):
    grounding: float = Field(ge=1.0, le=5.0, description="Fidelity to historical Apple support practices (1-5)")
    actionability: float = Field(ge=1.0, le=5.0, description="Clear diagnostics and concrete next steps (1-5)")
    tone: float = Field(ge=1.0, le=5.0, description="Apple brand voice: empathetic, professional, calm (1-5)")
    escalation: float = Field(ge=1.0, le=5.0, description="Correct routing decision and channel choice (1-5)")
    safety: float = Field(ge=1.0, le=5.0, description="PII preservation and physical safety warnings (1-5)")
    overall: float = Field(ge=1.0, le=5.0, description="Weighted composite score")
    critique: str = Field(default="", description="Qualitative feedback and rationale")


class EvaluationReport(BaseModel):
    total_samples: int
    intent_accuracy: float
    intent_macro_f1: float
    escalation_accuracy: float
    escalation_precision: float
    escalation_recall: float
    critical_safety_recall: float
    escalation_f1: float
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    bleu_score: float
    avg_judge_overall: float
    judge_human_kappa: Optional[float] = None
    judge_human_spearman: Optional[float] = None
    per_intent_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
