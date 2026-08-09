from __future__ import annotations

from importlib import import_module as _import_module


_EXPORT_MODULES = {
    "TRAINABLE_PREFIXES": ".contracts",
    "FreezeManifest": ".contracts",
    "TrainingBoundaryError": ".contracts",
    "assert_training_boundary": ".freeze",
    "build_optimizer": ".freeze",
    "configure_training_boundary": ".freeze",
    "extract_mutable_state": ".freeze",
}

__all__ = [
    "TRAINABLE_PREFIXES",
    "FreezeManifest",
    "TrainingBoundaryError",
    "assert_training_boundary",
    "build_optimizer",
    "configure_training_boundary",
    "extract_mutable_state",
]


def __getattr__(name: str):
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name)
        ) from error
    value = getattr(_import_module(module_name, __name__), name)
    globals()[name] = value
    return value
