from __future__ import annotations

from .base import DatasetPlugin


_plugins: dict[str, DatasetPlugin] = {}
_builtins_loaded = False


def register_dataset(plugin: DatasetPlugin) -> DatasetPlugin:
    names = (plugin.name, *plugin.aliases)
    duplicates = [name for name in names if name.lower() in _plugins]
    if duplicates:
        raise ValueError(f"Dataset names already registered: {duplicates}")
    for name in names:
        _plugins[name.lower()] = plugin
    return plugin


def _load_builtins() -> None:
    global _builtins_loaded
    if not _builtins_loaded:
        from . import builtins  # noqa: F401

        _builtins_loaded = True


def get_dataset(name: str) -> DatasetPlugin:
    _load_builtins()
    try:
        return _plugins[name.lower()]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset {name!r}; available: {', '.join(list_datasets())}"
        ) from exc


def list_datasets() -> list[str]:
    _load_builtins()
    return sorted({plugin.name for plugin in _plugins.values()})
