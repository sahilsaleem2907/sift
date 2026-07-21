"""Provider-agnostic POST /review endpoint.

Proves the GitHub contract is 100% backward-compatible (default provider, same
credential validation) and that unknown providers / bad credentials return 400.
No real token exchange, no real review — those boundaries are mocked.
"""
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.testclient import TestClient


def _build_app(monkeypatch):
    """Build the core app with a dummy DATABASE_URL so validate_required() passes.

    Also disable SIFT_API_KEY so /review does not require a bearer token in tests
    (the developer env may have one set).
    """
    monkeypatch.setattr("sift.config.DATABASE_URL", "postgresql://x")
    monkeypatch.setattr("sift.config.SIFT_API_KEY", None)
    from sift.api.app import build_app

    return build_app()


def test_build_app_registers_github_builder(monkeypatch) -> None:
    _build_app(monkeypatch)
    from sift.integrations.github_client import github_review_adapter
    from sift.integrations.registry import get_forge_builder

    assert get_forge_builder("github") is github_review_adapter


def test_review_backward_compat_github_token(monkeypatch) -> None:
    """Original GitHub body (no `provider`) still queues a review via the github adapter."""
    app = _build_app(monkeypatch)
    with (
        patch("sift.api.review.run_review", new=MagicMock()) as mock_run,
        patch(
            "sift.integrations.github_client.make_github_forge_builder",
            new=AsyncMock(return_value=lambda: object()),
        ),
    ):
        client = TestClient(app)
        r = client.post(
            "/review",
            json={"owner": "o", "repo": "r", "pr_number": 1, "github_token": "tok"},
        )
    assert r.status_code == 202
    assert mock_run.called


def test_review_installation_id_mode(monkeypatch) -> None:
    app = _build_app(monkeypatch)
    with (
        patch("sift.api.review.run_review", new=MagicMock()) as mock_run,
        patch(
            "sift.integrations.github_client.make_github_forge_builder",
            new=AsyncMock(return_value=lambda: object()),
        ),
    ):
        client = TestClient(app)
        r = client.post(
            "/review",
            json={"owner": "o", "repo": "r", "pr_number": 1, "installation_id": 42},
        )
    assert r.status_code == 202
    assert mock_run.called


def test_review_rejects_both_github_creds(monkeypatch) -> None:
    app = _build_app(monkeypatch)
    client = TestClient(app)
    r = client.post(
        "/review",
        json={
            "owner": "o", "repo": "r", "pr_number": 1,
            "github_token": "t", "installation_id": 5,
        },
    )
    assert r.status_code == 400


def test_review_rejects_neither_github_cred(monkeypatch) -> None:
    app = _build_app(monkeypatch)
    client = TestClient(app)
    r = client.post("/review", json={"owner": "o", "repo": "r", "pr_number": 1})
    assert r.status_code == 400


def test_review_unknown_provider(monkeypatch) -> None:
    app = _build_app(monkeypatch)
    client = TestClient(app)
    r = client.post(
        "/review",
        json={"provider": "nope", "owner": "o", "repo": "r", "pr_number": 1},
    )
    assert r.status_code == 400
