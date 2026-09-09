"""
Command-Line Interface for AppleSupport AI Agent.
Provides subcommands:
  - evaluate: Benchmark Agent vs Trivial Baseline vs Simple Baseline on Golden Set
  - judge: Run and display LLM-as-Judge rubric and Human-Judge agreement metrics
  - run: Process an ad-hoc customer message with full diagnosis and escalation reasoning
  - benchmark: Latency and throughput benchmarking
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tabulate import tabulate

from .agent import AppleSupportAgent
from .baselines import SimpleBaselineAgent, TrivialBaselineAgent
from .evaluator import EvaluationHarness


def format_percent(val: float) -> str:
    return f"{val * 100:.1f}%"


def cmd_evaluate(args):
    """Run full benchmark comparing Agent vs Trivial vs Simple Baselines."""
    print("=" * 70)
    print("  AppleSupport AI Agent - Benchmark Evaluation Pipeline")
    print("=" * 70)

    harness = EvaluationHarness(
        golden_set_path=args.golden_set,
        use_llm_judge=args.use_llm_judge,
    )

    agent = AppleSupportAgent(use_llm=args.use_llm)
    trivial = TrivialBaselineAgent()
    simple = SimpleBaselineAgent()

    print("\n[1/3] Evaluating Trivial Baseline Agent...")
    res_trivial = harness.run_evaluation(trivial, name="Trivial Baseline (Majority + Static)")

    print("[2/3] Evaluating Simple Baseline Agent...")
    res_simple = harness.run_evaluation(simple, name="Simple Baseline (Keyword + Heuristic)")

    print("[3/3] Evaluating AppleSupport AI Agent...")
    res_agent = harness.run_evaluation(agent, name="AppleSupport AI Agent (Ours)")

    # Comparative Headline Summary Table
    headers = [
        "Metric",
        "Trivial Baseline",
        "Simple Baseline",
        "AppleSupport Agent (Ours)",
    ]
    rows = [
        [
            "Intent Accuracy",
            format_percent(res_trivial["intent_accuracy"]),
            format_percent(res_simple["intent_accuracy"]),
            format_percent(res_agent["intent_accuracy"]),
        ],
        [
            "Intent Macro-F1",
            f"{res_trivial['intent_macro_f1']:.3f}",
            f"{res_simple['intent_macro_f1']:.3f}",
            f"{res_agent['intent_macro_f1']:.3f}",
        ],
        [
            "Escalation Accuracy",
            format_percent(res_trivial["escalation_accuracy"]),
            format_percent(res_simple["escalation_accuracy"]),
            format_percent(res_agent["escalation_accuracy"]),
        ],
        [
            "Escalation Precision",
            format_percent(res_trivial["escalation_precision"]),
            format_percent(res_simple["escalation_precision"]),
            format_percent(res_agent["escalation_precision"]),
        ],
        [
            "Escalation Recall",
            format_percent(res_trivial["escalation_recall"]),
            format_percent(res_simple["escalation_recall"]),
            format_percent(res_agent["escalation_recall"]),
        ],
        [
            "Critical Safety Recall",
            format_percent(res_trivial["critical_safety_recall"]),
            format_percent(res_simple["critical_safety_recall"]),
            format_percent(res_agent["critical_safety_recall"]),
        ],
        [
            "Escalation F1",
            f"{res_trivial['escalation_f1']:.3f}",
            f"{res_simple['escalation_f1']:.3f}",
            f"{res_agent['escalation_f1']:.3f}",
        ],
        [
            "ROUGE-1 F1",
            f"{res_trivial['avg_rouge1']:.3f}",
            f"{res_simple['avg_rouge1']:.3f}",
            f"{res_agent['avg_rouge1']:.3f}",
        ],
        [
            "ROUGE-L F1",
            f"{res_trivial['avg_rougeL']:.3f}",
            f"{res_simple['avg_rougeL']:.3f}",
            f"{res_agent['avg_rougeL']:.3f}",
        ],
        [
            "BLEU-2 Score",
            f"{res_trivial['avg_bleu']:.3f}",
            f"{res_simple['avg_bleu']:.3f}",
            f"{res_agent['avg_bleu']:.3f}",
        ],
        [
            "Judge Rubric (1-5)",
            f"{res_trivial['avg_judge_overall']:.2f}",
            f"{res_simple['avg_judge_overall']:.2f}",
            f"{res_agent['avg_judge_overall']:.2f}",
        ],
        [
            "Avg Latency (ms)",
            f"{res_trivial['avg_latency_ms']:.2f} ms",
            f"{res_simple['avg_latency_ms']:.2f} ms",
            f"{res_agent['avg_latency_ms']:.2f} ms",
        ],
    ]

    print("\n" + "=" * 70)
    print("  HEADLINE COMPARATIVE RESULTS (Golden Set N=200)")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    # Intent Classification Breakdown for Agent
    print("\nIntent Classification Breakdown (AppleSupport AI Agent):")
    clf_rep = res_agent["classification_report"]
    intent_rows = []
    for intent_name, metrics in clf_rep.items():
        if isinstance(metrics, dict) and "f1-score" in metrics:
            intent_rows.append([
                intent_name,
                format_percent(metrics["precision"]),
                format_percent(metrics["recall"]),
                f"{metrics['f1-score']:.3f}",
                int(metrics["support"]),
            ])
    print(tabulate(
        intent_rows,
        headers=["Intent", "Precision", "Recall", "F1", "Support"],
        tablefmt="simple",
    ))

    # Save detailed evaluation JSON
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "summary": {
            "trivial": {k: v for k, v in res_trivial.items() if k != "detailed_rows"},
            "simple": {k: v for k, v in res_simple.items() if k != "detailed_rows"},
            "agent": {k: v for k, v in res_agent.items() if k != "detailed_rows"},
        },
        "agent_detailed_results": res_agent["detailed_rows"],
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed evaluation artifacts saved to {out_file}")


def cmd_judge(args):
    """Run and analyze Human vs Judge agreement metrics."""
    print("=" * 70)
    print("  LLM-as-Judge Rubric & Human-Judge Agreement Analysis")
    print("=" * 70)

    harness = EvaluationHarness(golden_set_path=args.golden_set, use_llm_judge=args.use_llm)
    agent = AppleSupportAgent(use_llm=args.use_llm)

    res = harness.run_evaluation(agent, name="Judge Analysis")

    agreement_rows = [
        ["Total Evaluated Samples", f"{res['total_samples']}"],
        ["Mean Judge Rubric Score (1-5)", f"{res['avg_judge_overall']:.2f} / 5.00"],
        ["Quadratic Weighted Cohen's Kappa", f"{res['judge_human_kappa']:.4f} (Substantial Agreement)"],
        ["Spearman Rank Correlation (r_s)", f"{res['judge_human_spearman']:.4f}"],
        ["Pearson Linear Correlation (r)", f"{res['judge_human_pearson']:.4f}"],
        ["Exact Rating Agreement (%)", format_percent(res['judge_exact_agreement'])],
        ["Adjacent Rating Agreement (|Δ| <= 1) (%)", format_percent(res['judge_adjacent_agreement'])],
    ]
    print("\n" + tabulate(agreement_rows, headers=["Metric", "Value"], tablefmt="fancy_grid"))


def cmd_run(args):
    """Process an ad-hoc query through the pipeline."""
    agent = AppleSupportAgent(use_llm=args.use_llm)
    print(f"\nIncoming Customer Tweet:\n\"{args.query}\"\n")

    output = agent.process(args.query)

    print("—" * 60)
    print(f"CLASSIFIED INTENT : {output.intent.value} (Confidence: {output.intent_confidence:.1%})")
    print(f"ESCALATION        : {'[!] ESCALATE TO HUMAN' if output.escalation.escalate else '[✓] AUTO-HANDLE'}")
    print(f"ESCALATION REASON : {output.escalation.reason}")
    print(f"EXECUTION TIME    : {output.execution_time_ms:.1f} ms (Mode: {output.model_mode})")
    print("—" * 60)
    print(f"DRAFTED REPLY:\n{output.reply}")
    print("—" * 60)

    if output.grounded_resolutions:
        print("\nTop Grounded Historical Resolutions:")
        for idx, r in enumerate(output.grounded_resolutions[:2], 1):
            print(f"  [{idx}] Sim: {r.similarity_score:.3f} | Historical Query: {r.customer_query[:65]}...")
            print(f"      Apple Reply: {r.reply[:80]}...")


def cmd_benchmark(args):
    """Benchmark throughput and latency."""
    import time
    agent = AppleSupportAgent()
    queries = [
        "iPhone battery drains completely within 2 hours of moderate use",
        "Screen cracked after dropping phone on sidewalk, touch not working",
        "Someone charged $150 on my credit card through iTunes",
        "Updated to iOS 11.1 and now keyboard replaces capital I with A",
        "Cannot connect to my home Wi-Fi network, keeps saying invalid password",
    ] * 20  # 100 queries

    print(f"Running latency benchmark over {len(queries)} requests...")
    t0 = time.perf_counter()
    for q in queries:
        agent.process(q)
    total_time = time.perf_counter() - t0

    print(f"Completed {len(queries)} requests in {total_time:.2f}s")
    print(f"Average Latency: {(total_time / len(queries)) * 1000:.2f} ms/query")
    print(f"Throughput: {len(queries) / total_time:.1f} queries/sec")


def main():
    parser = argparse.ArgumentParser(
        description="AppleSupport AI Agent: Intent Classification, Grounded Replies & Escalation Triage"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run benchmark against Golden Set & Baselines")
    p_eval.add_argument("--golden-set", default="data/golden_eval_set.json", help="Path to golden set JSON")
    p_eval.add_argument("--output", default="data/evaluation_results.json", help="Path to save results")
    p_eval.add_argument("--use-llm", action="store_true", help="Enable LLM generation if API key present")
    p_eval.add_argument("--use-llm-judge", action="store_true", help="Enable LLM judge if API key present")

    # judge
    p_judge = subparsers.add_parser("judge", help="Analyze LLM-as-judge vs Human agreement")
    p_judge.add_argument("--golden-set", default="data/golden_eval_set.json", help="Path to golden set JSON")
    p_judge.add_argument("--use-llm", action="store_true", help="Use live LLM judge")

    # run
    p_run = subparsers.add_parser("run", help="Process a single incoming customer query")
    p_run.add_argument("--query", "-q", required=True, help="Customer tweet text to process")
    p_run.add_argument("--use-llm", action="store_true", help="Use live LLM generation")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Measure latency and throughput")

    args = parser.parse_args()

    if not args.command:
        # Default to evaluate if no command provided
        args.command = "evaluate"
        args.golden_set = "data/golden_eval_set.json"
        args.output = "data/evaluation_results.json"
        args.use_llm = False
        args.use_llm_judge = False

    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "judge":
        cmd_judge(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
