#!/usr/bin/env bash
# Full sentry run: fire Sift on ALL sentry (Python/Ruby) forked PRs sequentially, then print a
# per-PR + aggregate summary. Sibling of scripts/smoke_test.sh — where smoke_test validates one
# fork, this scales the same fire+poll harness to the whole sentry set for a scoreable pass.
#
# Fire-only: this does NOT clear prior comments and does NOT compute F1. Scoring is a separate
# pipeline in the benchmark repo (step1->step2->step3 --tool sift); commands are echoed at the end.
#
# Usage:
#   INSTALL_ID=<org installation id> bash scripts/full_sentry_run.sh
#
#   # Restrict to a subset (resume/retry) by original PR number:
#   FORKS="67876 95633" INSTALL_ID=... bash scripts/full_sentry_run.sh
#
# Prereqs: sift server already running (started with .env + bench_env.sh sourced), logging to $LOG.
set -uo pipefail

ORG="${ORG:-sift-benchmark}"
SIFT="${SIFT:-http://localhost:8000}"
LOG="${LOG:-$HOME/sift_smoke_sentry.log}"                           # file the running server tees to
INSTALL_ID="${INSTALL_ID:?set INSTALL_ID (org install id from the org settings UI)}"
BENCH_DIR="${BENCH_DIR:-$HOME/personal/code-review-benchmark/offline}"
MAX_WAIT="${MAX_WAIT:-2000}"   # seconds to wait per fork for the async review to finish
PR="${PR:-1}"                  # in-fork PR number (always 1 — the canonical code PR)
FORKS="${FORKS:-}"             # optional space-separated list of original PR #s; default = all
RESUME="${RESUME:-0}"          # 1 = if a fork's completion marker is already in $LOG, record it
                               #     from the log + posted comments WITHOUT re-firing (resume a run)

bot_comment_count() {  # $1=repo $2=pr -> count of sift-agent[bot] inline comments
  gh api "repos/$ORG/$1/pulls/$2/comments" \
    -q '[.[]|select(.user.login=="sift-agent[bot]")]|length' 2>/dev/null || echo 0
}

golden_count() {  # $1=original PR number -> golden issue count from sentry.json (or ?)
  python3 - "$BENCH_DIR/golden_comments/sentry.json" "$1" <<'PY' 2>/dev/null
import json, sys, re
path, pr = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path))
except Exception:
    print("?"); sys.exit()
for e in data:
    u = e.get("original_url") or e.get("url") or ""
    if re.search(rf"/{re.escape(pr)}(?:$|/)", u):
        print(len(e.get("comments") or [])); break
else:
    print("?")
PY
}

# --- preflight: server reachable + auth present -------------------------------
if ! curl -s -o /dev/null --max-time 10 "$SIFT/health" && ! curl -s -o /dev/null --max-time 10 "$SIFT"; then
  echo "ERROR: sift server not reachable at $SIFT (start it with .env + bench_env.sh sourced)"; exit 1
fi
AUTH_HEADER=()
if [ -n "${SIFT_API_KEY:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer $SIFT_API_KEY")
  echo "Auth: Bearer ${SIFT_API_KEY:0:6}…${SIFT_API_KEY: -4} (SIFT_API_KEY, ${#SIFT_API_KEY} chars)"
else
  echo "WARNING: SIFT_API_KEY unset — server will 401 if it requires auth"
fi

# --- discover sentry forks (config_prefix 'sentry' == all sentry PRs) ---------
ALL_REPOS=()   # bash 3.2 (macOS default) has no mapfile — read into the array manually
while IFS= read -r _line; do [ -n "$_line" ] && ALL_REPOS+=("$_line"); done < <(
  gh repo list "$ORG" --limit 400 --json name -q '.[].name' | grep -E '^sentry__.*__sift__' | sort)
[ "${#ALL_REPOS[@]}" -eq 0 ] && { echo "ERROR: no ^sentry__*__sift__ forks in org $ORG"; exit 1; }

REPOS=()
if [ -n "$FORKS" ]; then
  for r in "${ALL_REPOS[@]}"; do
    n=$(echo "$r" | sed -E 's/.*__PR([0-9]+)__.*/\1/')
    for want in $FORKS; do [ "$n" = "$want" ] && REPOS+=("$r"); done
  done
  [ "${#REPOS[@]}" -eq 0 ] && { echo "ERROR: FORKS='$FORKS' matched none of the discovered forks"; exit 1; }
else
  REPOS=("${ALL_REPOS[@]}")
fi
echo "Firing ${#REPOS[@]} fork(s) sequentially against $SIFT (MAX_WAIT=${MAX_WAIT}s each):"
printf '  %s\n' "${REPOS[@]}"

RUN_START=$(wc -c < "$LOG" 2>/dev/null || echo 0)   # log offset for the global health footer
ROWS=()   # "ORIG_PR|REPO|STATUS|AFTER|DELTA|GOLD|ELAPSED"
n_done=0; n_failed=0; n_timeout=0; n_skip=0

# --- per-fork loop ------------------------------------------------------------
for REPO in "${REPOS[@]}"; do
  ORIG_PR=$(echo "$REPO" | sed -E 's/.*__PR([0-9]+)__.*/\1/')
  echo
  echo "=== $ORG/$REPO PR#$PR (original sentry #$ORIG_PR) ==="

  STATE=$(gh pr view "$PR" --repo "$ORG/$REPO" --json state -q '.state' 2>/dev/null)
  if [ "$STATE" != "OPEN" ]; then
    echo "  SKIP: PR#$PR not open (state=${STATE:-missing})"
    ROWS+=("$ORIG_PR|$REPO|SKIP|-|-|-|-"); n_skip=$((n_skip+1)); continue
  fi

  # RESUME: if this fork already completed in the current $LOG, record it without re-firing.
  if [ "$RESUME" = "1" ] && grep -qE "Review (completed|failed) for $ORG/$REPO PR #$PR" "$LOG" 2>/dev/null; then
    if grep -qE "Review failed for $ORG/$REPO PR #$PR" "$LOG"; then status="FAILED"; else status="DONE"; fi
    AFTER=$(bot_comment_count "$REPO" "$PR"); GOLD=$(golden_count "$ORIG_PR"); GOLD="${GOLD:-?}"
    echo "  RESUME: already $status in log — posted=$AFTER golden=$GOLD (not re-fired)"
    ROWS+=("$ORIG_PR|$REPO|${status}*|$AFTER|-|$GOLD|cached")
    case "$status" in DONE) n_done=$((n_done+1));; FAILED) n_failed=$((n_failed+1));; esac
    continue
  fi

  BEFORE=$(bot_comment_count "$REPO" "$PR")
  OFF=$(wc -c < "$LOG" 2>/dev/null || echo 0)
  echo "  baseline sift-agent[bot] comments: $BEFORE"
  echo "  firing POST $SIFT/review ..."
  curl -s -X POST "$SIFT/review" -H 'Content-Type: application/json' \
    ${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"} \
    -d "{\"owner\":\"$ORG\",\"repo\":\"$REPO\",\"pr_number\":$PR,\"installation_id\":$INSTALL_ID}"
  echo

  # Poll the log tail (past OFF) for this fork's completion/failure marker.
  status="TIMEOUT"; waited=0
  while [ "$waited" -lt "$MAX_WAIT" ]; do
    sleep 15; waited=$((waited+15))
    slice=$(tail -c +$((OFF+1)) "$LOG" 2>/dev/null)
    if grep -qE "Review failed for $ORG/$REPO PR #$PR" <<<"$slice"; then
      status="FAILED"; break
    fi
    if grep -qE "Review completed for $ORG/$REPO PR #$PR" <<<"$slice"; then
      status="DONE"; break
    fi
    echo "  ... ${waited}s"
  done

  AFTER=$(bot_comment_count "$REPO" "$PR")
  GOLD=$(golden_count "$ORIG_PR"); GOLD="${GOLD:-?}"
  DELTA=$((AFTER - BEFORE))
  echo "  -> $status in ${waited}s | posted=$AFTER (Δ=+$DELTA) | golden=$GOLD"
  ROWS+=("$ORIG_PR|$REPO|$status|$AFTER|+$DELTA|$GOLD|${waited}s")
  case "$status" in
    DONE) n_done=$((n_done+1));; FAILED) n_failed=$((n_failed+1));; TIMEOUT) n_timeout=$((n_timeout+1));;
  esac
done

# --- summary table ------------------------------------------------------------
echo
echo "================ FULL SENTRY RUN SUMMARY ================"
printf '%-8s %-46s %-8s %-8s %-7s %-6s %-7s\n' "PR#" "fork" "status" "posted" "Δ" "golden" "elapsed"
printf '%.0s-' {1..96}; echo
for row in "${ROWS[@]}"; do
  IFS='|' read -r pr repo st af dl gd el <<<"$row"
  printf '%-8s %-46s %-8s %-8s %-7s %-6s %-7s\n' "$pr" "$repo" "$st" "$af" "$dl" "$gd" "$el"
done
printf '%.0s-' {1..96}; echo
echo "totals: DONE=$n_done  FAILED=$n_failed  TIMEOUT=$n_timeout  SKIP=$n_skip"

# --- global health footer (cumulative over this run's log slice) --------------
SLICE=$(tail -c +$((RUN_START+1)) "$LOG" 2>/dev/null)
echo
echo "--- run health (over this run's log slice) ---"
echo "  parse FAILURE count (want 0; nonzero => deepseek-style broken tool loop): $(grep -c 'parse FAILURE' <<<"$SLICE")"
auth_hits=$(grep -icE '401|invalid_api_key|unauthorized' <<<"$SLICE")
echo "  auth errors (401/invalid_api_key/unauthorized): $auth_hits"
grep -qiE 'auto-promot.*codeql|codeql.*finding|\[static_promote\].*codeql|Running CodeQL' <<<"$SLICE" \
  && echo "  CodeQL: fired ✓" || echo "  CodeQL: no lines (Skipping/absent)"
grep -qE '\[Vector\]' <<<"$SLICE" \
  && echo "  Vector DB: emitting ✓" || echo "  Vector DB: no [Vector] lines (Ollama down / pgvector unset?)"

# --- next steps: clearing + scoring (separate pipeline) -----------------------
cat <<EOF

--- to score (separate pipeline; needs MARTIAN_API_KEY) ---
This script never clears comments. Before a scored pass, clear any fork that carries stale
sift-agent[bot] comments (Δ smaller than 'posted' above => pre-existing comments), e.g.:
  gh api repos/$ORG/<REPO>/pulls/$PR/comments \\
    -q '.[]|select(.user.login=="sift-agent[bot]").id' |
    xargs -I{} gh api -X DELETE repos/$ORG/<REPO>/pulls/comments/{}
  # and issue comments: repos/$ORG/<REPO>/issues/$PR/comments -> issues/comments/{id}

Then, from $BENCH_DIR (its own .venv):
  uv run python -m code_review_benchmark.step1_download_prs --output results/benchmark_data.json --force --tool sift
  uv run python -m code_review_benchmark.step2_extract_comments --tool sift
  uv run python -m code_review_benchmark.step2_5_dedup_candidates --tool sift
  uv run python -m code_review_benchmark.step3_judge_comments --tool sift --dedup-groups results/<judge-model>/dedup_groups.json
========================================================
EOF
