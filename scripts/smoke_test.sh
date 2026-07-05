#!/usr/bin/env bash
# Smoke test: fire Sift on ONE interpreted-language (Sentry/Python) forked PR, then
# print a 5-check verdict. Validates LLM wiring, bot identity, CodeQL, vector DB, and
# precision sanity before committing to the full 50-PR run.
#
# Usage:
#   INSTALL_ID=<org installation id> BENCH_DIR=~/personal/code-review-benchmark/offline \
#     bash scripts/smoke_test.sh
#
# Prereqs: sift server running (started with bench_env.sh overlay), logging to $LOG.
set -uo pipefail

ORG="${ORG:-sift-benchmark}"
SIFT="${SIFT:-http://localhost:8000}"
LOG="${LOG:-$HOME/sift_smoke.log}"
INSTALL_ID="${INSTALL_ID:?set INSTALL_ID (org install id from the org settings UI)}"
BENCH_DIR="${BENCH_DIR:-$HOME/personal/code-review-benchmark/offline}"
MAX_WAIT="${MAX_WAIT:-2000}"   # seconds to wait for the async review to post

bot_comment_count() {  # $1=repo $2=pr -> count of sift-agent[bot] inline comments
  gh api "repos/$ORG/$1/pulls/$2/comments" \
    -q '[.[]|select(.user.login=="sift-agent[bot]")]|length' 2>/dev/null || echo 0
}

# --- 1. pick a Sentry fork + its canonical code PR (PR#1) ----------------------
REPO=$(gh repo list "$ORG" --limit 200 --json name -q '.[].name' | grep -i '__sentry__' | head -1)
[ -z "$REPO" ] && { echo "ERROR: no '__sentry__' fork found in org $ORG"; exit 1; }
PR=1   # PR#1 is the canonical code PR in every fork (PR#2+ are Dependabot noise)
STATE=$(gh pr view "$PR" --repo "$ORG/$REPO" --json state -q '.state' 2>/dev/null)
[ "$STATE" = "OPEN" ] || { echo "ERROR: $ORG/$REPO PR#$PR not open (state=${STATE:-missing})"; exit 1; }
echo "Smoke target: $ORG/$REPO PR#$PR"

# --- 2. fire the review (202 async) --------------------------------------------
BEFORE=$(bot_comment_count "$REPO" "$PR")
echo "Baseline sift-agent[bot] comments: $BEFORE"
echo "Firing POST $SIFT/review ..."
AUTH_HEADER=()
if [ -n "${SIFT_API_KEY:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer $SIFT_API_KEY")
  echo "Auth: Bearer ${SIFT_API_KEY:0:6}…${SIFT_API_KEY: -4} (SIFT_API_KEY, ${#SIFT_API_KEY} chars)"
else
  echo "Auth: none (SIFT_API_KEY unset — works only if server has no SIFT_API_KEY)"
fi
curl -s -X POST "$SIFT/review" -H 'Content-Type: application/json' \
  ${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"} \
  -d "{\"owner\":\"$ORG\",\"repo\":\"$REPO\",\"pr_number\":$PR,\"installation_id\":$INSTALL_ID}"
echo

# --- 3. poll until comments appear ---------------------------------------------
waited=0
while [ "$waited" -lt "$MAX_WAIT" ]; do
  sleep 15; waited=$((waited+15))
  now=$(bot_comment_count "$REPO" "$PR")
  if [ "$now" -gt "$BEFORE" ]; then echo "  ✓ $now comments after ${waited}s"; break; fi
  echo "  ... ${waited}s, still $now comments"
done
AFTER=$(bot_comment_count "$REPO" "$PR")

# --- 4. verdict ----------------------------------------------------------------
echo
echo "================ SMOKE TEST VERDICT ================"

echo "--- [1] LLM wiring: DeepSeek parse failures (want 0) + auth errors ---"
echo "  parse FAILURE count: $(grep -c 'parse FAILURE' "$LOG" 2>/dev/null || echo 0)"
grep -iE '401|invalid_api_key|rate.?limit|unauthorized' "$LOG" 2>/dev/null | tail -3 || echo "  (no auth errors)"

echo "--- [2] Bot identity: comments by sift-agent[bot] (want >0) ---"
echo "  $ORG/$REPO PR#$PR -> $AFTER bot comments"

echo "--- [3] CodeQL: fired with findings? (Python path needs no build) ---"
grep -iE 'auto-promot.*codeql|codeql.*finding|\[static_promote\].*codeql|Running CodeQL|Skipping CodeQL' "$LOG" 2>/dev/null | tail -8 || echo "  (no CodeQL log lines)"

echo "--- [4] Vector DB: emitting similar-snippet blocks, or silent no-op? ---"
grep -E '\[Vector\]' "$LOG" 2>/dev/null | tail -6 || echo "  (no [Vector] lines — likely Ollama down / pgvector unset)"

echo "--- [5] Precision sanity: sift posted vs golden-issue count ---"
ORIG_PR=$(echo "$REPO" | sed -E 's/.*__PR([0-9]+)__.*/\1/')
GOLD=$(python3 - "$BENCH_DIR/golden_comments/sentry.json" "$ORIG_PR" <<'PY' 2>/dev/null
import json, sys, re
path, pr = sys.argv[1], sys.argv[2]
data = json.load(open(path))
for e in data:
    u = e.get("original_url") or e.get("url") or ""
    if re.search(rf"/{re.escape(pr)}(?:$|/)", u):
        print(len(e.get("comments") or [])); break
else:
    print("?")
PY
)
echo "  original Sentry PR #$ORIG_PR  |  golden issues: ${GOLD:-?}  |  sift posted: $AFTER"
echo "  (healthy ~1-2x golden; a 4x+ blow-out = FP flood hurting precision/F1)"
echo "===================================================="
