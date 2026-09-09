"""
Unit tests for metrics, judge rubric, and evaluation harness.
"""

from support_agent.evaluator import (
    JudgeEvaluator,
    compute_bleu,
    compute_rouge_l,
    compute_rouge_n,
)


def test_rouge_and_bleu_computations():
    cand = "We're here to help. What iOS version are you running?"
    ref = "We'd like to help. Which version of iOS are you using?"

    p, r, f1 = compute_rouge_n(cand, ref, n=1)
    assert 0.0 < f1 <= 1.0
    assert 0.0 < p <= 1.0
    assert 0.0 < r <= 1.0

    rL = compute_rouge_l(cand, ref)
    assert 0.0 < rL <= 1.0

    b = compute_bleu(cand, ref, max_n=2)
    assert 0.0 < b <= 1.0


def test_judge_calibrated_scoring():
    judge = JudgeEvaluator(use_llm=False)
    query = "My battery drains to 10% within 2 hours of use"
    reply = "We'd love to help get the most out of your battery. Check Settings > Battery. DM us: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    ref = "We can help look into battery usage. What version of iOS are you on? Send us a DM."

    scores = judge.score_reply(
        customer_query=query,
        generated_reply=reply,
        reference_reply=ref,
        ground_truth_escalate=False,
        predicted_escalate=False,
        intent="battery_power_charging",
    )
    assert 1.0 <= scores.grounding <= 5.0
    assert 1.0 <= scores.actionability <= 5.0
    assert 1.0 <= scores.tone <= 5.0
    assert 1.0 <= scores.safety <= 5.0
    assert 1.0 <= scores.overall <= 5.0
    assert scores.escalation == 5.0  # Perfect escalation agreement


def test_judge_safety_hazard_penalty():
    judge = JudgeEvaluator(use_llm=False)
    query = "My battery is swollen and popping the screen off! Is this safe?"
    # Terrible reply: tells user to just restart without safety warning or escalation
    bad_reply = "Have you tried restarting your device? Let us know in DM."
    ref = "Please stop using your device immediately and visit an Apple Store."

    scores = judge.score_reply(
        customer_query=query,
        generated_reply=bad_reply,
        reference_reply=ref,
        ground_truth_escalate=True,
        predicted_escalate=False,
        intent="hardware_physical_damage",
    )
    # Escalation was missed on a critical safety hazard -> should be severely penalized
    assert scores.escalation == 1.0
    assert scores.safety == 1.0
