"""Unit tests for the forge-builder registry (provider → builder factory)."""
import pytest

from sift.integrations.registry import get_forge_builder, register_forge_builder


async def _fake_factory(body):  # pragma: no cover - trivial
    return lambda: object()


def test_register_and_get_forge_builder() -> None:
    register_forge_builder("dummy_provider", _fake_factory)
    assert get_forge_builder("dummy_provider") is _fake_factory


def test_get_forge_builder_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_forge_builder("no_such_provider")
