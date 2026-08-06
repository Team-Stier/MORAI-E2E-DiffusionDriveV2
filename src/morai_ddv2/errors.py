from __future__ import annotations


class MoraiDDV2Error(Exception):
    """Base exception for deterministic MORAI preprocessing."""


class ConfigurationError(MoraiDDV2Error):
    """Raised when a domain configuration violates its schema or contract."""


class IntegrityError(MoraiDDV2Error):
    """Raised when immutable input or generated artifact identity is invalid."""


class SensorPresetError(ConfigurationError):
    """Raised when the pinned MORAI sensor preset cannot be interpreted."""
