#!/usr/bin/env bash
# Install CodeQL bundle: download and extract to /opt/codeql.
# Expects CODEQL_BUNDLE_URL (full URL) or CODEQL_BUNDLE_VERSION (e.g. 2.14.0) in env.
# Exits non-zero on failure. Idempotent: skips if /opt/codeql/codeql already exists.
set -euo pipefail

CODEQL_DIR="/opt/codeql"
if [[ -x "${CODEQL_DIR}/codeql" ]]; then
  echo "CodeQL already present at ${CODEQL_DIR}, skipping."
  exit 0
fi

if [[ -z "${CODEQL_BUNDLE_URL:-}" ]]; then
  VERSION="${CODEQL_BUNDLE_VERSION:-2.16.6}"
  # Remove leading 'v' if present to avoid double 'v' in the URL
  VERSION="${VERSION#v}"
  CODEQL_BUNDLE_URL="https://github.com/github/codeql-action/releases/download/codeql-bundle-v${VERSION}/codeql-bundle-linux64.tar.gz"
fi

echo "Downloading CodeQL bundle from ${CODEQL_BUNDLE_URL}..."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -sSLf "$CODEQL_BUNDLE_URL" -o "$TMP/bundle.tar.gz"
tar -xzf "$TMP/bundle.tar.gz" -C "$TMP"
# Tarball has single top-level dir (e.g. codeql or codeql-bundle-linux64)
TOP=$(ls -1 "$TMP" | grep -v bundle.tar | head -1)
if [[ -z "$TOP" ]] || [[ ! -d "$TMP/$TOP" ]]; then
  echo "Unexpected bundle layout" >&2
  exit 1
fi
mkdir -p /opt
mv "$TMP/$TOP" "$CODEQL_DIR"
chmod -R a+rX "$CODEQL_DIR"
if [[ ! -x "${CODEQL_DIR}/codeql" ]]; then
  echo "CodeQL binary not found at ${CODEQL_DIR}/codeql" >&2
  exit 1
fi
echo "CodeQL installed at ${CODEQL_DIR}."
