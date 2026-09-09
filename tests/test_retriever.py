"""
Unit tests for HistoricalRetriever and Canonical Protocols.
"""

import pytest
from support_agent.retriever import HistoricalRetriever
from support_agent.schemas import Intent


@pytest.fixture(scope="module")
def retriever():
    return HistoricalRetriever()


def test_retriever_protocol(retriever):
    proto = retriever.get_protocol(Intent.BATTERY_POWER_CHARGING)
    assert "Settings > Battery" in proto
    assert "Battery & Power" in proto


def test_retriever_search(retriever):
    results = retriever.retrieve("iPhone battery dying fast after charging", top_k=3)
    assert len(results) > 0
    assert results[0].similarity_score >= 0.0
    assert isinstance(results[0].customer_query, str)
    assert isinstance(results[0].reply, str)


def test_retriever_with_intent_filter(retriever):
    results = retriever.retrieve(
        "Screen cracked and broken",
        top_k=2,
        intent_filter=Intent.HARDWARE_PHYSICAL_DAMAGE,
    )
    assert len(results) > 0
