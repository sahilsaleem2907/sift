#!/usr/bin/env bash
# Fire Sift on all keycloak forks — per-project sibling of full_sentry_run.sh.
#
# Thin wrapper over full_benchmark_run.sh with PROJECTS pinned to keycloak, so you can run and
# analyse one project's PRs at a time. All the multi-repo harness features pass through via env:
#
#   INSTALL_ID=<org install id> bash scripts/full_keycloak_run.sh
#   FORKS="123 456" INSTALL_ID=... bash scripts/full_keycloak_run.sh   # subset by in-name PR #
#   RESUME=1 INSTALL_ID=... bash scripts/full_keycloak_run.sh          # record from log, don't re-fire
#
# Prereqs: sift server running with .env + scripts/bench_env.sh sourced (Opus+Sonnet, high effort).
exec env PROJECTS="keycloak" bash "$(dirname "$0")/full_benchmark_run.sh" "$@"
