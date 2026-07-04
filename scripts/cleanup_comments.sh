#!/usr/bin/env bash
# Revert: delete sift-agent[bot] comments from forked benchmark PRs so a later run
# (or a different model config) starts from a clean slate. Deletes both inline review
# comments and the "## Sift Review" summary issue comment.
#
# Usage:
#   # one repo:
#   ORG=sift-benchmark bash scripts/cleanup_comments.sh <repo-name>
#   # all forks in the org:
#   ORG=sift-benchmark ALL=1 bash scripts/cleanup_comments.sh
set -uo pipefail

ORG="${ORG:-sift-benchmark}"
BOT="sift-agent[bot]"

clean_repo() {  # $1 = repo name
  local repo="$1"
  local pr=1   # PR#1 is the canonical code PR; clean it whatever its state (comments persist post-merge)
  gh pr view "$pr" --repo "$ORG/$repo" --json number >/dev/null 2>&1 \
    || { echo "skip $repo (PR#$pr not found)"; return; }

  local inline summary n=0
  # inline review comments
  for id in $(gh api "repos/$ORG/$repo/pulls/$pr/comments" \
      -q ".[]|select(.user.login==\"$BOT\")|.id" 2>/dev/null); do
    gh api -X DELETE "repos/$ORG/$repo/pulls/comments/$id" >/dev/null && n=$((n+1))
  done
  # summary issue comment
  for id in $(gh api "repos/$ORG/$repo/issues/$pr/comments" \
      -q ".[]|select(.user.login==\"$BOT\")|.id" 2>/dev/null); do
    gh api -X DELETE "repos/$ORG/$repo/issues/comments/$id" >/dev/null && n=$((n+1))
  done
  echo "  $repo PR#$pr -> deleted $n bot comment(s)"
}

if [ "${ALL:-0}" = "1" ]; then
  for repo in $(gh repo list "$ORG" --limit 200 --json name -q '.[].name'); do
    clean_repo "$repo"
  done
else
  [ $# -lt 1 ] && { echo "usage: ORG=$ORG bash scripts/cleanup_comments.sh <repo>  (or ALL=1 for every fork)"; exit 1; }
  clean_repo "$1"
fi
echo "cleanup done"
