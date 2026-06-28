"""Root pytest configuration.

Hooks here run before any test module or source module is imported,
making them the correct place to suppress third-party import-time warnings
that pyproject.toml filterwarnings cannot catch (those apply only during
the test-execution phase, not the collection/import phase).
"""

import warnings


def pytest_configure(config: object) -> None:  # noqa: ARG001
    """Register warning filters before any module is imported."""
    # holidays v0.x emits FutureIncompatibilityWarning about v1.0 at import time.
    # Filter by module origin (uses re.match on __name__) — avoids message text
    # matching issues caused by the warning's leading newline character.
    warnings.filterwarnings("ignore", module=r"holidays")
