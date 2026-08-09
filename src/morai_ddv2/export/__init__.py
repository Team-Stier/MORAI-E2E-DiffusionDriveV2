from __future__ import annotations

from importlib import import_module as _import_module


_EXPORT_MODULES = {
    "SelSchemaReport": ".schema",
    "TensorSchemaEntry": ".state_hash",
    "assert_only_allowed_changed": ".schema",
    "merge_mutable_state": ".schema",
    "state_dict_sha256": ".state_hash",
    "state_schema_sha256": ".state_hash",
    "tensor_data_sha256": ".state_hash",
    "validate_raw_schema": ".schema",
}

__all__ = [
    "SelSchemaReport",
    "TensorSchemaEntry",
    "assert_only_allowed_changed",
    "merge_mutable_state",
    "state_dict_sha256",
    "state_schema_sha256",
    "tensor_data_sha256",
    "validate_raw_schema",
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
