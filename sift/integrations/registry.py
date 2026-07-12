"""Forge provider registry: register and look up ForgeProvider implementations."""
from typing import Any, Awaitable, Callable, Dict, Type

from sift.integrations.base import ForgeProvider

_registry: Dict[str, Type[ForgeProvider]] = {}


def register_forge(key: str, cls: Type[ForgeProvider]) -> None:
    """Register a ForgeProvider implementation under *key* (e.g. 'github', 'bitbucket')."""
    _registry[key] = cls


def get_forge(key: str) -> Type[ForgeProvider]:
    """Return the ForgeProvider class registered under *key*.

    Raises KeyError with a descriptive message if the key is unknown.
    """
    if key not in _registry:
        known = ", ".join(sorted(_registry)) or "(none registered)"
        raise KeyError(f"No forge registered for key {key!r}. Known forges: {known}")
    return _registry[key]


# -- forge-builder registry --
#
# A *builder factory* resolves provider-specific credentials from a review request
# and returns a zero-arg callable that yields a ready-to-enter ForgeProvider. This
# lets the provider-agnostic POST /review endpoint dispatch to any provider without
# importing it. The request argument is duck-typed (kept as Any) so this module does
# not depend on the api layer's request model.
ForgeBuilder = Callable[[], ForgeProvider]
ForgeBuilderFactory = Callable[[Any], Awaitable[ForgeBuilder]]

_builder_registry: Dict[str, ForgeBuilderFactory] = {}


def register_forge_builder(provider: str, factory: ForgeBuilderFactory) -> None:
    """Register a builder factory under *provider* (e.g. 'github', 'bitbucket').

    GitHub self-registers in ``build_app``; enterprise providers register in their
    own composition entrypoint (e.g. ``sift_enterprise.main``).
    """
    _builder_registry[provider] = factory


def get_forge_builder(provider: str) -> ForgeBuilderFactory:
    """Return the builder factory registered under *provider*.

    Raises KeyError with a descriptive message if the provider is unknown.
    """
    if provider not in _builder_registry:
        known = ", ".join(sorted(_builder_registry)) or "(none registered)"
        raise KeyError(
            f"No forge builder registered for provider {provider!r}. Known providers: {known}"
        )
    return _builder_registry[provider]
