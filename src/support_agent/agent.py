"""
AppleSupportAgent: Unified orchestrator combining intent classification,
grounded historical retrieval, escalation policy engine, and response drafting.
"""

import time
from typing import Optional

from .escalation_engine import EscalationPolicyEngine
from .intent_classifier import IntentClassifier
from .retriever import HistoricalRetriever
from .response_generator import ResponseGenerator
from .schemas import AgentOutput


class AppleSupportAgent:
    """
    Production-grade AI Support Agent for AppleSupport.
    Classifies incoming customer messages into 7 distinct intents,
    retrieves grounded historical resolution patterns, decides whether
    to auto-handle or escalate with a stated reason, and drafts an authentic reply.
    """

    def __init__(
        self,
        classifier_path: Optional[str] = "data/processed/intent_classifier.joblib",
        kb_path: str = "data/processed/apple_knowledge_base.json",
        use_llm: bool = False,
        model_name: str = "gpt-4o-mini",
    ):
        self.use_llm = use_llm
        self.model_name = model_name
        self.classifier = IntentClassifier(model_path=classifier_path, use_llm=use_llm)
        self.retriever = HistoricalRetriever(kb_path=kb_path)
        self.escalation_engine = EscalationPolicyEngine()
        self.response_generator = ResponseGenerator(model_name=model_name, use_llm=use_llm)

    def process(self, customer_query: str) -> AgentOutput:
        """
        Process a single customer message through the full agentic pipeline.
        """
        t0 = time.perf_counter()

        # Step 1: Intent Classification
        classification = self.classifier.predict(customer_query)

        # Step 2: Grounded Historical Retrieval
        grounded_resolutions = self.retriever.retrieve(
            query=customer_query,
            top_k=3,
            intent_filter=classification.intent,
        )
        protocol = self.retriever.get_protocol(classification.intent)

        # Step 3: Escalation Decision Engine (Audited Reason)
        escalation_decision = self.escalation_engine.evaluate(
            query=customer_query,
            classification=classification,
        )

        # Step 4: Grounded Reply Generation
        reply = self.response_generator.generate(
            query=customer_query,
            intent=classification.intent,
            escalation=escalation_decision,
            grounded_resolutions=grounded_resolutions,
            protocol=protocol,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        return AgentOutput(
            customer_query=customer_query,
            intent=classification.intent,
            intent_confidence=classification.confidence,
            escalation=escalation_decision,
            reply=reply,
            grounded_resolutions=grounded_resolutions,
            suggested_protocol=protocol,
            execution_time_ms=round(latency_ms, 2),
            model_mode="llm" if (self.use_llm and self.response_generator.has_openai) else "offline",
        )
