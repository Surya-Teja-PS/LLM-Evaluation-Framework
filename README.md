# LLM Evaluation Framework — Study Assistant

An end-to-end evaluation pipeline for benchmarking AI persona responses across multiple quality dimensions, combining a custom LLM-as-judge scorer with DeepEval's automated metrics for cross-validation.

## Overview

This project evaluates a "Study Assistant" chatbot that answers ML/AI questions in two different personas — **Friendly** and **Academic** — and scores each response along 6 quality criteria using two independent judging methods. The goal is not just to score responses, but to check whether a manual LLM-as-judge and an automated metric framework (DeepEval) actually *agree* with each other.

## Why This Exists

Most single-method eval setups trust one judge's score at face value. This framework was built to answer a more useful question: **does a hand-written judging prompt agree with an established, automated evaluation library?** If they disagree often, that's a signal the manual judge (or the automated one) is unreliable. If they agree, you can trust either one going forward with more confidence.

## Architecture

```
Question + Persona
        │
        ▼
  Study Assistant (Groq, Llama 3.1-8b-instant)
        │
        ▼
   AI Response
        │
   ┌────┴─────┐
   ▼          ▼
Manual Judge   DeepEval Metrics
(Groq,        (AnswerRelevancyMetric,
Llama 3.3-70b) GEval — Accuracy)
   │              │
   └──────┬───────┘
          ▼
   Aggregated Results (CSV)
          │
          ▼
 Summary Stats + Persona Comparison Chart
```

## Test Dataset

- **10 questions** spanning `easy` / `medium` / `hard` difficulty (e.g., "What are LLMs?" → "Explain the attention mechanism in transformers")
- **2 personas** — Friendly (encouraging, beginner-facing) and Academic (formal, precise)
- **20 total test cases** (10 questions × 2 personas) by design

## Scoring Methods

### 1. Manual LLM-as-Judge
A Groq-hosted judge model (`llama-3.3-70b-versatile`) scores each response 1–5 on:
- Accuracy
- Clarity
- Relevance
- Use of Analogies
- Follow-up Question
- Persona Consistency

### 2. DeepEval Automated Metrics
- **`AnswerRelevancyMetric`** — does the response actually address the question?
- **`GEval` (Accuracy)** — custom criteria-based scoring for factual correctness

Both DeepEval metrics run on a Groq-backed judge model (`GroqDeepEvalModel`) instead of Gemini, to avoid Gemini's tighter free-tier quota.

### 3. Cross-Method Agreement
The pipeline compares the manual judge's accuracy score (≥4 = "pass") against DeepEval's `GEval` pass/fail to compute an **agreement rate** — the core validation metric of this project.

## Engineering Decisions Worth Noting

- **Unified retry logic across two providers.** Groq and Gemini raise different exception types for the same underlying failures (rate limits, server overload) — `RateLimitError`/`APIStatusError` on Groq vs. `ClientError`/`ServerError` on Gemini. `call_with_retry()` normalizes handling for both with provider-appropriate backoff.
- **Key collision fix.** The manual judge's `PERSONA` score is stored under `persona_consistency`, explicitly kept separate from the `persona` field (Friendly/Academic label) to prevent one from silently overwriting the other in the results dict.
- **Per-case checkpointing.** Results are written to CSV after every single test case, not just at the end — a crash or rate limit mid-run doesn't cost you completed work.
- **JSON retry handling for DeepEval.** Groq occasionally returns malformed JSON where Gemini's structured output mode wouldn't — `run_deepeval()` retries a few times before failing, since regenerating usually fixes it.

## Results (Sample Run — 15/20 cases completed)

| Metric | Value |
|---|---|
| Avg. Accuracy | 5.00 / 5 |
| Avg. Clarity | 4.60 / 5 |
| Avg. Relevance | 5.00 / 5 |
| Avg. Analogies | 4.87 / 5 |
| Avg. Follow-up | 4.93 / 5 |
| Avg. Persona Consistency | 4.93 / 5 |
| DeepEval Relevancy Pass Rate | 93.3% |
| DeepEval Accuracy Pass Rate | 100% |
| Manual vs. GEval Agreement | 100% |

> Note: this run completed 15 of the 20 designed test cases before stopping (likely a rate limit). The checkpointing design means all 15 completed cases were saved regardless.

## Tech Stack

- **Language:** Python
- **LLM APIs:** Groq (Llama 3.1-8b-instant, Llama 3.3-70b-versatile), Gemini
- **Evaluation:** DeepEval (`AnswerRelevancyMetric`, `GEval`)
- **Data/Analysis:** pandas, matplotlib

## Project Structure

```
llm_eval_framework.py     # Full pipeline: assistant, judges, retry logic, aggregation
eval_results.csv          # Final results after a run
eval_results_partial.csv  # Live checkpoint file, updated after every test case
persona_comparison.png    # Bar chart: Friendly vs Academic across all 6 criteria
```

## Running It

```bash
pip install -U groq google-genai deepeval pandas matplotlib
python llm_eval_framework.py
```

Requires `GROQ_API_KEY` and `GEMINI_API_KEY` environment variables (or Colab secrets, as configured in the script).

## Possible Extensions

- Complete the remaining 5 test cases to get a full 20-case sample
- Add a third, independent judge to strengthen the cross-validation design
- Track cost/latency per API call alongside quality scores
- Expand the test set beyond ML/AI topics to check generalization
