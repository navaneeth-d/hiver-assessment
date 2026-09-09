"""
Grounded Historical Resolution Retriever for AppleSupport AI Agent.
Indexes historical resolved customer queries and Apple responses, performing
semantic/lexical retrieval to ground replies in actual Apple support practices.
"""

import json
from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import Intent, RetrievedResolution


CANONICAL_PROTOCOLS = {
    Intent.BATTERY_POWER_CHARGING: (
        "Protocol [Battery & Power]:\n"
        "- Gather: Device model & iOS version (Settings > General > About).\n"
        "- Diagnostics: Review Settings > Battery for top consuming apps and Battery Health (Maximum Capacity).\n"
        "- Troubleshooting: Restart device, disable Background App Refresh for heavy apps, avoid charging in extreme heat.\n"
        "- Routing: If battery health < 80% or unexpected shutdowns occur, direct to DM or Apple Authorized Service for battery replacement."
    ),
    Intent.SOFTWARE_UPDATE_OS: (
        "Protocol [iOS & System Performance]:\n"
        "- Gather: Exact iOS build (Settings > General > About).\n"
        "- Diagnostics: Verify available storage in Settings > General > iPhone Storage (minimum 5-10 GB recommended).\n"
        "- Troubleshooting: Perform forced restart (volume up, volume down, hold side button). For known text/keyboard anomalies, check Text Replacement or update to latest release.\n"
        "- Routing: If device is in boot loop or frozen on Apple logo, guide to iTunes/Finder recovery mode."
    ),
    Intent.HARDWARE_PHYSICAL_DAMAGE: (
        "Protocol [Hardware & Physical Safety]:\n"
        "- Safety Warning: If battery is swollen or casing is separated, stop using and charging immediately to prevent thermal hazard.\n"
        "- Physical Assessment: Cracked glass or water immersion requires hardware diagnostic.\n"
        "- Troubleshooting: Do NOT place in rice (dust damages ports); back up data if display is partially responsive.\n"
        "- Routing: Immediate human escalation; schedule in-person Genius Bar appointment or Authorized Service Provider visit."
    ),
    Intent.ACCOUNT_SECURITY_BILLING: (
        "Protocol [Apple ID, Security & Purchases]:\n"
        "- Security Warning: Never post Apple ID passwords, credit card numbers, or 2FA codes publicly.\n"
        "- Self-Service Tools: For password/lockout, direct to iforgot.apple.com; for purchases and refund disputes, direct to reportaproblem.apple.com.\n"
        "- Routing: Must escalate to private DM or human account specialist for identity verification."
    ),
    Intent.CONNECTIVITY_NETWORK: (
        "Protocol [Wi-Fi, Bluetooth & Cellular]:\n"
        "- Troubleshooting: 1. Toggle Airplane mode on for 15s then off. 2. Forget Wi-Fi network and reconnect. 3. Settings > General > Reset > Reset Network Settings.\n"
        "- Diagnostics: Check for carrier update via Settings > General > About.\n"
        "- Routing: If cellular says 'No Service' consistently or Wi-Fi toggle is greyed out, DM for diagnostic check."
    ),
    Intent.APPS_MEDIA_FEATURES: (
        "Protocol [App Crashes & Media Sync]:\n"
        "- Troubleshooting: 1. Force close the app and restart phone. 2. Check App Store > Updates for the latest app build. 3. Delete and reinstall the app.\n"
        "- Media/iCloud: For Apple Music/Photos sync, verify Apple ID login in Settings and check apple.com/support/systemstatus."
    ),
    Intent.STORE_ORDERS_RESERVATIONS: (
        "Protocol [Orders, Deliveries & Genius Bar]:\n"
        "- Tracking: Direct customer to check apple.com/orderstatus with their W-order number.\n"
        "- In-Store: Direct to apple.com/retail to view Genius Bar reservations and store hours.\n"
        "- Routing: If courier marks delivered but package is missing, escalate to human support for carrier claims investigation."
    ),
}


class HistoricalRetriever:
    """
    Retrieves the most semantically relevant historical AppleSupport resolutions
    to ground agent replies in authentic brand history.
    """

    def __init__(self, kb_path: str = "data/processed/apple_knowledge_base.json"):
        self.kb_path = Path(kb_path)
        self.documents: List[dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors = None
        self._load_and_index()

    def _load_and_index(self):
        if not self.kb_path.exists():
            print(f"Retriever warning: {self.kb_path} not found. Running with empty KB.")
            return

        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        if not self.documents:
            return

        # Index customer query text + reply context
        corpus = [doc["customer_text"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=25000,
            stop_words="english",
        )
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        intent_filter: Optional[Intent] = None,
    ) -> List[RetrievedResolution]:
        """Find the top-k most similar historical resolutions for the query."""
        if not self.documents or self.vectorizer is None or self.doc_vectors is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        # If intent filter provided, boost documents matching the intent
        intent_val = intent_filter.value if intent_filter else None

        scored_indices = []
        for idx, score in enumerate(similarities):
            doc = self.documents[idx]
            final_score = float(score)
            if intent_val and doc.get("intent") == intent_val:
                final_score *= 1.25  # Intent alignment boost
            scored_indices.append((idx, final_score))

        scored_indices.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_indices[:top_k]:
            raw_intent = doc.get("intent")
            intent_str = "general_support"
            if isinstance(raw_intent, str) and raw_intent.strip():
                intent_str = raw_intent.strip()
            results.append(
                RetrievedResolution(
                    id=str(doc.get("id", str(idx))),
                    customer_query=str(doc.get("customer_text", "")),
                    reply=str(doc.get("reply", "")),
                    intent=intent_str,
                    similarity_score=round(score, 4),
                )
            )

        return results

    def get_protocol(self, intent: Intent) -> str:
        """Fetch the official Apple troubleshooting protocol for an intent."""
        return CANONICAL_PROTOCOLS.get(intent, "Standard diagnostic inquiry.")
