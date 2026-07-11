# Sift eval harness

Offline golden-set scorer for the review pipeline. Runs without GitHub.

## Setup

From the repo root, with dependencies installed (`pip install -r requirements.txt`):

```bash
export PYTHONPATH=.
```

## Run

```bash
python -m eval.run_eval --model ollama/llama3.2 --effort low
python -m eval.run_eval --model anthropic/claude-3-5-sonnet-20241022 --effort balanced
python -m eval.run_eval --case 001_null_deref --effort low
```

Requires a reachable LLM (LiteLLM). Set `LLM_API_BASE` / API keys as for the main app.

## Metrics

- **Precision** — matched expected findings / total findings emitted
- **Recall** — matched expected findings / total expected
- **Noise-rate** — findings with no expected match (excluding declared false-positive lines)

## Cases

Synthetic diffs under `eval/cases/` with planted bugs. Add curated real PRs later as JSON + diff pairs.
