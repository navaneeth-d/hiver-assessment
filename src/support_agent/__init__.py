"""
AppleSupport AI Agent Package.
"""

from .agent import AppleSupportAgent
from .baselines import SimpleBaselineAgent, TrivialBaselineAgent
from .cli import main
from .escalation_engine import EscalationPolicyEngine
from .evaluator import EvaluationHarness, JudgeEvaluator
from .intent_classifier import IntentClassifier
from .retriever import HistoricalRetriever
from .schemas import (
    AgentOutput,
    EscalationDecision,
    EscalationReasonCategory,
    Intent,
    IntentClassificationResult,
    RetrievedResolution,
    RubricScores,
)

__all__ = [
    "AppleSupportAgent",
    "TrivialBaselineAgent",
    "SimpleBaselineAgent",
    "IntentClassifier",
    "HistoricalRetriever",
    "EscalationPolicyEngine",
    "EvaluationHarness",
    "JudgeEvaluator",
    "Intent",
    "EscalationDecision",
    "EscalationReasonCategory",
    "IntentClassificationResult",
    "RetrievedResolution",
    "RubricScores",
    "AgentOutput",
    "main",
]
