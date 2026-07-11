"""Forge provider registry: register and look up ForgeProvider implementations."""
from typing import Dict, Type

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
