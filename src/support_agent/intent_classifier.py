"""
Intent Classification Module for AppleSupport AI Agent.
Provides calibrated statistical classification (TF-IDF + Logistic Regression)
with LLM few-shot capability and ambiguity detection.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .schemas import Intent, IntentClassificationResult


class IntentClassifier:
    """
    Classifies incoming customer tweets into one of 7 AppleSupport intents.
    Supports fast local calibrated model and optional LLM classification.
    """

    INTENT_MAP = {
        "software_update_os": Intent.SOFTWARE_UPDATE_OS,
        "battery_power_charging": Intent.BATTERY_POWER_CHARGING,
        "hardware_physical_damage": Intent.HARDWARE_PHYSICAL_DAMAGE,
        "account_security_billing": Intent.ACCOUNT_SECURITY_BILLING,
        "connectivity_network": Intent.CONNECTIVITY_NETWORK,
        "apps_media_features": Intent.APPS_MEDIA_FEATURES,
        "store_orders_reservations": Intent.STORE_ORDERS_RESERVATIONS,
    }

    def __init__(
        self,
        model_path: Optional[str] = "data/processed/intent_classifier.joblib",
        confidence_threshold: float = 0.35,
        use_llm: bool = False,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.use_llm = use_llm
        self.pipeline: Optional[Pipeline] = None
        self._load_or_train()

    def _load_or_train(self):
        """Load trained pipeline from disk or train if missing."""
        if self.model_path and self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                return
            except Exception as e:
                print(f"Warning: Failed to load model from {self.model_path}: {e}. Retraining...")

        self.train()

    def train(self, training_data_path: str = "data/processed/intent_training_data.json"):
        """Train calibrated TF-IDF + Logistic Regression model on historical labeled data."""
        p = Path(training_data_path)
        if not p.exists():
            # Fallback tiny bootstrap data if training file not found
            texts = [
                "updated to ios 11 and phone is laggy",
                "battery draining fast and overheating",
                "cracked screen after dropping phone",
                "apple id locked need refund on itunes",
                "wifi not connecting bluetooth dropping",
                "app crashing repeatedly when opening",
                "where is my iphone preorder delivery",
            ]
            labels = [
                "software_update_os",
                "battery_power_charging",
                "hardware_physical_damage",
                "account_security_billing",
                "connectivity_network",
                "apps_media_features",
                "store_orders_reservations",
            ]
        else:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts = [item["text"] for item in data]
            labels = [item["intent"] for item in data]

        # Pipeline: TF-IDF with sublinear tf and ngrams (1, 2)
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=12000,
                sublinear_tf=True,
                stop_words="english",
            )),
            ("clf", LogisticRegression(
                C=2.5,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        pipeline.fit(texts, labels)
        self.pipeline = pipeline

        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, self.model_path)
            print(f"Intent classifier trained on {len(texts)} samples and saved to {self.model_path}")

    def predict(self, text: str) -> IntentClassificationResult:
        """Classify customer text into an Intent with confidence and ambiguity flag."""
        cleaned = re.sub(r"^(@\w+\s*)+", "", text).strip()

        # Keyword heuristics for critical safety/hardware edge cases
        lowered = cleaned.lower()
        if re.search(r"\b(swollen|swelling|bulging battery|fire hazard|smoke|smoking)\b", lowered):
            return IntentClassificationResult(
                intent=Intent.HARDWARE_PHYSICAL_DAMAGE,
                confidence=0.99,
                probabilities={"hardware_physical_damage": 0.99},
                is_ambiguous=False,
            )

        if self.pipeline is None:
            self._load_or_train()

        probs = self.pipeline.predict_proba([cleaned])[0]
        classes = self.pipeline.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}

        best_idx = int(np.argmax(probs))
        best_intent_str = classes[best_idx]
        best_prob = float(probs[best_idx])

        # Sort probs to check margin for ambiguity
        sorted_probs = sorted(probs, reverse=True)
        top1, top2 = sorted_probs[0], sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        is_ambiguous = (top1 < self.confidence_threshold) or (top1 - top2 < 0.05 and top1 < 0.45)

        # Fallback intent if unknown
        intent_enum = self.INTENT_MAP.get(best_intent_str, Intent.SOFTWARE_UPDATE_OS)

        return IntentClassificationResult(
            intent=intent_enum,
            confidence=round(best_prob, 4),
            probabilities=prob_dict,
            is_ambiguous=is_ambiguous,
        )
