"""Python interface for e3nn-core-rs fixed-weight MLIP inference."""

from __future__ import annotations

import os
import sys

# Baked Apple-Silicon builds are distributed for the Metal-enabled MACE path.
# Keep this as a default rather than an override so callers can still opt out
# with LLMLIP_MACE_METAL=0 before importing llmlip.
if sys.platform == "darwin":
    os.environ.setdefault("LLMLIP_MACE_METAL", "1")

try:
    from ._llmlip import (
        BackendUnavailableError,
        CliBackendError,
        LlmlipError,
        InvalidStructureError,
        Model,
        ModelCapabilities,
        ModelLoadError,
        Prediction,
        PredictionError,
        PredictionInputRequirements,
        UnsupportedCapabilityError,
    )
    _NATIVE_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _NATIVE_IMPORT_ERROR = exc

    class LlmlipError(Exception):
        """Base exception for llmlip Python bindings."""

    class ModelLoadError(LlmlipError):
        """Raised when a model cannot be loaded."""

    class PredictionError(LlmlipError):
        """Raised when prediction fails."""

    class UnsupportedCapabilityError(LlmlipError):
        """Raised when a requested model capability is unsupported."""

    class InvalidStructureError(LlmlipError):
        """Raised when structure inputs are invalid."""

    class BackendUnavailableError(LlmlipError):
        """Raised when no requested backend is available."""

    class CliBackendError(LlmlipError):
        """Raised when the CLI backend fails."""

    class ModelCapabilities:  # type: ignore[no-redef]
        pass

    class PredictionInputRequirements:  # type: ignore[no-redef]
        pass

    class Prediction:  # type: ignore[no-redef]
        pass

    class Model:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise BackendUnavailableError(
                "llmlip native module is unavailable"
            ) from _NATIVE_IMPORT_ERROR

        @classmethod
        def load(cls, *args, **kwargs):
            return cls(*args, **kwargs)


def load_baked_model() -> Model:
    """Load the baked MACE-OAM Medium model bundled in this extension."""

    try:
        return Model.load_baked_mace_oam_medium()
    except AttributeError as exc:
        raise BackendUnavailableError(
            "this llmlip build does not include the baked MACE-OAM Medium model"
        ) from exc


__all__ = [
    "BackendUnavailableError",
    "CliBackendError",
    "LlmlipError",
    "InvalidStructureError",
    "load_baked_model",
    "Model",
    "ModelCapabilities",
    "ModelLoadError",
    "Prediction",
    "PredictionError",
    "PredictionInputRequirements",
    "UnsupportedCapabilityError",
]
