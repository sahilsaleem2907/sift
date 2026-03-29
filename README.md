<p align="center">
  <img src="./assets/logo.svg" alt="Sift" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/sahilsaleem2907/sift"><img src="https://img.shields.io/github/stars/sahilsaleem2907/sift?style=flat-square" alt="GitHub stars"/></a>
  <img src="https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"/>
  <a href="https://github.com/sahilsaleem2907/sift/pkgs/container/sift"><img src="https://img.shields.io/badge/GHCR-package-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub Container Registry"/></a>
  <img src="https://img.shields.io/badge/open%20source-%E2%9C%93-22c55e?style=flat-square" alt="Open source"/>
</p>

# Sift

**Sift** is an open-source backend for **AI-assisted pull request review**: it combines static analysis (Semgrep, language linters, optional CodeQL), smart routing, and LLM-powered reasoning via [LiteLLM](https://github.com/BerriAI/litellm). Run it yourself with Docker, wire it to GitHub (Actions or webhooks), and keep configuration in environment variables or a hosted setup flow.

---

## Get started

| Path | Use when |
|------|----------|
| **[Website](https://YOUR_WEBSITE)** (replace with your URL) | You want a guided UI to configure Sift and related settings. |
| **[Documentation](https://YOUR_DOCS)** (replace with your URL) | You want the full reference for options, API, and operations. |
| **This repository** | You prefer to self-host from source, build images yourself, or integrate manually. |

---

## Requirements

- **PostgreSQL** (connection string via `DATABASE_URL`)
- An **LLM** reachable by LiteLLM (local Ollama, OpenAI, Anthropic, Gemini, Azure, Bedrock, etc.—see [.env.example](.env.example))
- For GitHub features: either **`SWIFT_API_BACKEND_BASE_URL`** (installation token service) or **`SIFT_GITHUB_TOKEN`**, as described in [src/config.py](src/config.py)

---

## Docker (GHCR)

Publish or pull an image matching your registry layout. Example image reference (adjust org and tag if yours differ):

```text
ghcr.io/sahilsaleem2907/sift:latest
```

Build targets in the [Dockerfile](Dockerfile):

| Target | Role |
|--------|------|
| **`slim`** | Smaller image: Python app, common apt linters (e.g. shellcheck, yamllint). Build with `--target slim` when you want a lighter image. |
| **`full`** | Default when you run `docker build` with no `--target`: adds CodeQL and a broad set of language linters (larger image, slower build). |

**Build locally**

```bash
docker build --target slim -t sift:slim .
docker build --target full -t sift:full .
```

**Run** (minimal; extend `-e` flags from [.env.example](.env.example) and your LLM provider)

```bash
docker run --rm -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:password@host:5432/sift" \
  -e LLM_MODEL="ollama/llama3.2" \
  -e LLM_API_BASE="http://host.docker.internal:11434" \
  sift:slim
```

The API listens on **port 8000**. Health check: `GET /health`.

---

## Local development

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set DATABASE_URL and LLM settings
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

---

## GitHub Actions

This repo ships a **reusable workflow** that pulls your container image, runs it with secrets, waits for `/health`, then calls `POST /review`. See [.github/workflows/sift-review.yml](.github/workflows/sift-review.yml).

Required secrets for the workflow include **`SIFT_IMAGE`** and **`SIFT_DATABASE_URL`**; others (API keys, LLM, GitHub App installation, etc.) are optional depending on your auth mode. Client repositories call this workflow with `workflow_call` on `pull_request` events.

---

## API overview

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness and database connectivity |
| `POST /review` | Trigger a PR review (body: `owner`, `repo`, `pr_number`, optional `before_sha`, `github_token` or `installation_id`) |

Details belong in your **documentation** site once you replace the placeholder link above.

---

## Configuration

Environment variables are documented in [.env.example](.env.example) (LLM, Semgrep, CodeQL, embeddings, concurrency, and more).

---

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/sahilsaleem2907/sift).

---

## License

This project is intended to be open source. Add a `LICENSE` file at the repository root when you choose a specific license, then you can add a matching badge to this README.
