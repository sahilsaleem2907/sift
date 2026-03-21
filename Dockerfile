# --- Version pins (single source of truth) ---
ARG CODEQL_BUNDLE_VERSION=v2.16.6
ARG HADOLINT_VERSION=v2.12.0
ARG TFLINT_VERSION=v0.50.0

# --- Stage 1: build ---
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: slim (minimal runtime + apt linters only) ---
FROM python:3.12-slim AS slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libpq5 shellcheck cppcheck yamllint lua-check \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY . .
ENV SWIFT_API_BACKEND_BASE_URL=https://528a-122-172-85-82.ngrok-free.app
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Stage 3: full (slim + CodeQL + all linters via scripts) ---
FROM slim AS full
ENV SWIFT_API_BACKEND_BASE_URL=https://528a-122-172-85-82.ngrok-free.app
ARG CODEQL_BUNDLE_VERSION
ARG HADOLINT_VERSION
ARG TFLINT_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
    zstd curl ca-certificates unzip \
    nodejs npm golang-go default-jdk ruby ruby-dev make gcc rustc mono-mcs php-cli perl r-base cpanminus \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/install-codeql.sh scripts/install-linters.sh /tmp/
RUN chmod +x /tmp/install-codeql.sh /tmp/install-linters.sh
RUN CODEQL_BUNDLE_VERSION="${CODEQL_BUNDLE_VERSION}" /tmp/install-codeql.sh && rm /tmp/install-codeql.sh
ENV PATH="/opt/codeql/codeql:/opt/codeql:$PATH"
RUN HADOLINT_VERSION="${HADOLINT_VERSION}" TFLINT_VERSION="${TFLINT_VERSION}" /tmp/install-linters.sh && rm /tmp/install-linters.sh
ENV PATH="/root/.composer/vendor/bin:$PATH"
