# AppleSupport AI Agent

This project implements an AI customer-support agent for AppleSupport using real
customer-support conversations from the Customer Support on Twitter (TWCS)
dataset.

The agent performs the three required tasks:

1. Classifies each incoming customer message into an intent learned from the data.
2. Drafts a reply grounded in historically observed AppleSupport resolutions.
3. Decides whether to auto-handle the message or escalate it to a human, with a reason.

The default pipeline is deterministic and offline. An LLM is optional and is not
needed to reproduce the headline results.

## Scope and Problem Framing

### What good means for AppleSupport

For this project, a trustworthy support agent should:

- classify common Apple support issues consistently;
- provide relevant, actionable troubleshooting grounded in historical replies;
- never silently auto-handle safety-critical hardware, account-security, billing,
  logistics, or severe-distress cases; and
- expose its escalation decision and reason so that the decision can be audited.

Safety and escalation correctness matter more than producing a fluent reply. The
agent therefore uses ordered policy rules for high-risk cases and a deterministic
offline response generator for reproducibility.

### What is intentionally not built

This is a support-triage and reply-drafting prototype, not a production support
system. It does not authenticate customers, access Apple order or account systems,
process refunds, verify identity, send messages, diagnose hardware remotely, or
replace human support. The historical data is noisy and the inferred intent labels
are bootstrap labels rather than official Apple taxonomy.

## Reproduce the Results

Requirements:

- Python 3.13 or newer
- `uv` package manager
- The repository checkout, including the prepared files under `data/processed/`
  and `data/golden_eval_set.json`

From the repository root:

```bash
uv sync
uv run support_agent evaluate --output data/evaluation_results.json
```

The command runs the agent against the 200-example golden set, compares it with
both baselines, prints the headline tables, and writes detailed results to
`data/evaluation_results.json`. With the committed prepared artifacts and offline
defaults, this is the reproducible path intended to complete in under 15 minutes.

Run the test suite with:

```bash
uv run pytest
```

The executable can also be invoked directly after `uv sync`:

```bash
support_agent evaluate
```

All paths in the CLI are relative to the repository root. Run commands from that
directory.

## Usage

```bash
# Full comparison against the golden set
support_agent evaluate

# Process one message through the complete pipeline
support_agent run --query "My iPhone battery is swollen and bulging!"

# Measure local inference latency and throughput over 100 messages
support_agent benchmark

# Analyze judge scores and agreement with human calibration scores
support_agent judge
```

The default mode is offline. Optional LLM generation and LLM judging can be
enabled with `--use-llm` and `--use-llm-judge`; these require a supported API key
such as `OPENAI_API_KEY` and make results dependent on external service behavior.

## Data and Preparation

The primary data source is the Kaggle **Customer Support on Twitter (TWCS)**
dataset, which contains roughly three million noisy tweets, multi-turn threads,
and conversations with many brands. This project selects AppleSupport as the
brand. The optional Banking77 dataset was not used.

**Data citation:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter), Kaggle. The implementation uses
the open-source Python libraries declared in [pyproject.toml](pyproject.toml).

The prepared artifacts are:

```text
data/
├── golden_eval_set.json
├── processed/
│   ├── apple_knowledge_base.json
│   ├── intent_classifier.joblib
│   └── intent_training_data.json
└── twcs/twcs.csv
```

To regenerate the processed artifacts from the raw TWCS CSV:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/build_golden_set.py
```

`prepare_data.py` reads up to 1,500,000 CSV rows, extracts customer messages
paired with AppleSupport replies, keeps root conversation starters, applies
high-precision heuristic intent labels, and writes up to 6,000 knowledge-base
pairs plus the training data. The script expects the raw file at
`data/twcs/twcs.csv`.

`build_golden_set.py` uses `random.seed(42)`, adds curated edge cases, and samples
natural TWCS examples into exact intent quotas for a total of 200 examples. The
golden set contains labels for intent, escalation, escalation reason, reference
reply, difficulty, and human rubric scores. It is kept separate from the training
and retrieval artifacts.

### Golden-set composition

| Intent | Examples |
|---|---:|
| `software_update_os` | 38 |
| `battery_power_charging` | 32 |
| `hardware_physical_damage` | 26 |
| `account_security_billing` | 32 |
| `connectivity_network` | 26 |
| `apps_media_features` | 26 |
| `store_orders_reservations` | 20 |
| **Total** | **200** |

The set includes curated edge cases covering swollen batteries, water damage,
account takeover, unauthorized charges, legal threats, stolen deliveries,
underspecified messages, multi-intent symptoms, sarcasm, and extreme frustration.
The resulting split is 80% auto-handle and 20% escalate.

## System Design

`AppleSupportAgent.process()` in [agent.py](src/hiver_assessment/agent.py) runs
four stages:

1. **Intent classification**: a TF-IDF unigram/bigram vectorizer and balanced
   Logistic Regression model over seven intents. Safety keywords such as swollen
   or smoking batteries override the model with a high-confidence hardware label.
2. **Grounded retrieval**: TF-IDF cosine similarity over historical customer
   queries, with a 25% boost for documents matching the predicted intent. Each
   intent also has a canonical troubleshooting protocol.
3. **Escalation policy**: ordered safety-first rules for hazards, physical damage,
   account compromise, billing, lockout/2FA, lost deliveries, distress/legal
   threats, ambiguity, and standard self-service.
4. **Reply generation**: deterministic templates use the escalation result,
   retrieved resolutions, and protocol. Replies include the AppleSupport DM link.

The seven intents are:

```text
software_update_os
battery_power_charging
hardware_physical_damage
account_security_billing
connectivity_network
apps_media_features
store_orders_reservations
```

## Evaluation

The evaluation harness in [evaluator.py](src/hiver_assessment/evaluator.py) runs
the same golden set through the production agent and two baselines.

### Automated metrics

- Intent accuracy and macro-F1
- Escalation accuracy, precision, recall, and F1
- Critical safety recall for safety-hardware examples
- ROUGE-1, ROUGE-2, ROUGE-L, and sentence BLEU-2 against reference replies
- Average per-example latency
- Per-intent precision, recall, F1, and support

### Reply-quality judge

Each reply receives 1-5 scores for grounding, actionability, tone, escalation
appropriateness, and safety. The weighted overall score is:

```text
25% grounding + 25% actionability + 15% tone
+ 20% escalation + 15% safety
```

By default this is a deterministic, calibrated rule-based judge, so the benchmark
does not require an API key. With `--use-llm-judge`, the harness can call an LLM
judge instead.

The human calibration scores stored in the golden set are compared with judge
scores using quadratic weighted Cohen's kappa, Pearson correlation, Spearman
correlation, exact agreement, and adjacent agreement.

### Baselines

| Baseline | Intent | Escalation | Reply |
|---|---|---|---|
| Trivial | Always majority class `software_update_os` | Never escalates | Static greeting |
| Simple | Naive keyword matching | A few escalation keywords | Generic diagnostic prompt |

These baselines establish whether the full pipeline improves on both a trivial
majority strategy and a lightweight heuristic system.

## Results

Run `support_agent evaluate` to generate the current result table and the
machine-readable artifact at `data/evaluation_results.json`. The table compares
the two baselines with the AppleSupport agent on intent, escalation, grounding,
judge, and latency metrics.

The verified offline run in this repository produced:

| Metric | Trivial | Simple | AppleSupport |
|---|---:|---:|---:|
| Intent accuracy | 19.0% | 57.0% | **98.5%** |
| Intent macro-F1 | 0.046 | 0.571 | **0.986** |
| Escalation accuracy | 80.0% | 80.5% | **94.5%** |
| Escalation F1 | 0.000 | 0.049 | **0.864** |
| Critical safety recall | 0.0% | 0.0% | **100.0%** |
| Judge rubric | 3.63 | 3.80 | **4.56** |
| Average latency | 0.10 ms | 0.50 ms | **1.75 ms** |

The complete report also includes ROUGE/BLEU scores, per-intent metrics, and
judge-human agreement statistics.

The headline results should be read as performance on this fixed, stratified,
partly curated 200-example set. They are not a production accuracy estimate and
do not measure real customer outcomes.

## Failure Analysis

The following are representative failure modes exposed by the golden set and the
design of the pipeline:

1. **Sparse or ambiguous messages**: “it just won't work anymore :(” has no device,
   symptom, or context. A forced intent is inherently unreliable; the intended
   mitigation is ambiguity detection and human escalation.
2. **Multi-intent messages**: a post-update message can mention both battery drain
   and a disabled Wi-Fi toggle. The single-label taxonomy may prioritize the wrong
   symptom even when the escalation decision is reasonable.
3. **Noisy inferred labels**: rule-based labels from TWCS can misclassify messages
   containing overloaded words such as “charge,” “store,” or “account.” The
   mitigation is curated review of the golden set, but training labels remain
   imperfect.
4. **Lexical retrieval mismatch**: semantically similar issues expressed with
   different vocabulary may retrieve a weak historical reply. TF-IDF retrieval is
   lexical rather than a true semantic embedding model.
5. **Policy-language mismatch**: sarcasm, profanity, or distress can express a
   serious situation without the exact escalation keywords. For example, the
   keyboard/autocorrect complaint uses hostile language but is still a standard
   software issue. Keyword ordering and curated edge cases reduce, but do not
   eliminate, these errors.

## What Is Misleading About the Headline Number?

The headline results can overstate real-world performance because the golden set
is only 200 examples, has exact intent quotas, includes curated edge cases, and
uses heuristic labels for much of its naturally sampled portion. The data is from
one brand, historical replies are not necessarily ideal replies, and the test
examples are not an independent random sample of future customer traffic.

The judge score is also not a human outcome metric: the default judge is a set of
rules calibrated against human rubric values, and ROUGE/BLEU reward lexical overlap
with one reference reply even when another reply would be better. Latency excludes
startup and indexing because the benchmark times only repeated `process()` calls.
LLM-enabled runs introduce further model and service variability.

## What I Would Do With One More Week

- Have a second reviewer independently label a larger, held-out test set and
  measure inter-rater disagreement.
- Replace heuristic labels and TF-IDF retrieval with reviewed labels and a hybrid
  sparse-plus-embedding retriever.
- Add confidence calibration and abstention thresholds measured against safety
  recall, rather than selecting thresholds manually.
- Test temporal drift, unseen phrasing, and cross-intent messages explicitly.
- Evaluate with blinded human pairwise preferences and safety red-team cases.
- Add structured observability, privacy filtering, and authenticated handoff
  contracts before any production integration.

## Decision Log

1. Chose AppleSupport because the TWCS data contains enough recognizable Apple
   support patterns for grounded reply generation.
2. Defined seven useful intents from the observed conversations instead of
   importing an external taxonomy.
3. Used TF-IDF plus Logistic Regression for fast, reproducible, offline inference.
4. Added a safety override for swollen batteries and related thermal hazards.
5. Used ordered escalation rules so safety checks run before general frustration.
6. Evaluated lost-delivery rules before generic distress rules to preserve logistics
   routing.
7. Used intent boosting during retrieval to reduce cross-topic historical matches.
8. Added canonical protocols so replies do not depend entirely on one retrieved
   historical message.
9. Kept the LLM optional so headline results are reproducible without credentials.
10. Used a deterministic calibrated judge by default to make evaluation repeatable.
11. Created a 200-example stratified golden set with exact intent quotas.
12. Added curated safety, security, logistics, ambiguity, and distress cases because
    aggregate accuracy alone is insufficient for a support agent.
13. Compared against both a majority/static baseline and a keyword heuristic
    baseline to establish meaningful lower bounds.
14. Reported safety recall and escalation metrics separately from reply similarity
    because a fluent reply can still make an unsafe routing decision.

## Repository Layout

```text
src/hiver_assessment/
├── agent.py                 # Four-stage pipeline orchestrator
├── intent_classifier.py     # TF-IDF + Logistic Regression
├── retriever.py             # Historical retrieval and protocols
├── escalation_engine.py     # Audited escalation policy
├── response_generator.py    # Offline and optional LLM replies
├── baselines.py             # Trivial and simple baselines
├── evaluator.py             # Metrics and judge agreement
├── schemas.py               # Pydantic models and enums
└── cli.py                   # evaluate, judge, run, benchmark
scripts/                     # Data preparation and golden-set generation
tests/                       # Unit and end-to-end tests
```

## Submission

Submit the repository link and this report through the assignment form:

https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f?pvs=105

The repository may be public or private with access granted to the evaluators.
Do not email the submission. Borrowed datasets, libraries, and techniques should
be cited; this project uses the TWCS dataset and standard open-source Python
libraries listed in `pyproject.toml`.