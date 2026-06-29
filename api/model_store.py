"""Singleton container for the Production model loaded at API startup.

Populated by api.loader.load_production_models() during lifespan.
Defaults to an empty (not_ready) state so TestClient without lifespan works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelStore:
    """Holds the Production ensemble model and its registry metadata.

    Attributes:
        model:         mlflow.pyfunc.PyFuncModel, or None if not yet loaded.
        model_version: Registry version string (e.g. "3"), or "not_loaded".
        trained_at:    ISO 8601 UTC timestamp of when the version was
                       registered (e.g. "2026-01-15T08:00:00Z"), or None.
    """

    model: Optional[Any] = None
    model_version: str = "not_loaded"
    trained_at: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """True when a Production model has been successfully loaded."""
        return self.model is not None
