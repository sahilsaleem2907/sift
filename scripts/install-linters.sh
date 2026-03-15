#!/usr/bin/env bash
# Install linter CLIs: hadolint, tflint, npm globals, rubocop, phpstan, Perl::Critic, R lintr.
# Expects HADOLINT_VERSION (e.g. v2.12.0), TFLINT_VERSION (e.g. v0.50.0) in env.
# Exits non-zero on failure.
set -euo pipefail

HADOLINT_VERSION="${HADOLINT_VERSION:-v2.12.0}"
TFLINT_VERSION="${TFLINT_VERSION:-v0.50.0}"

# --- Hadolint ---
echo "Installing hadolint ${HADOLINT_VERSION}..."
curl -sSLf "https://github.com/hadolint/hadolint/releases/download/${HADOLINT_VERSION}/hadolint-Linux-x86_64" -o /usr/local/bin/hadolint
chmod +x /usr/local/bin/hadolint

# --- TFLint ---
echo "Installing tflint ${TFLINT_VERSION}..."
TFLINT_URL="https://github.com/terraform-linters/tflint/releases/download/${TFLINT_VERSION}/tflint_linux_amd64.zip"
curl -sSLf "$TFLINT_URL" -o /tmp/tflint.zip
unzip -q -o /tmp/tflint.zip -d /usr/local/bin tflint
chmod +x /usr/local/bin/tflint
rm -f /tmp/tflint.zip

# --- Node global linters ---
echo "Installing npm global linters..."
npm install -g eslint typescript stylelint markdownlint-cli

# --- Ruby rubocop ---
echo "Installing rubocop..."
gem install rubocop --no-document

# --- PHP Composer + phpstan ---
echo "Installing composer and phpstan..."
curl -sSf https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer
composer global require phpstan/phpstan --no-interaction --prefer-dist

# --- Perl::Critic ---
echo "Installing Perl::Critic..."
cpanm --notest Perl::Critic

# --- R lintr ---
echo "Installing R lintr..."
R -e "install.packages('lintr', repos='https://cloud.r-project.org', quiet=TRUE)"

echo "Linters installed."
