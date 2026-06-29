"""Root pytest configuration.

holidays.deprecations.v1_incompatibility fires FutureIncompatibilityWarning
at MODULE IMPORT TIME via a bare warnings.warn() call. Every filter-based
approach fails because pytest's own "always::DeprecationWarning" mechanism
overrides user filterwarnings during its warning capture context.

The reliable fix: pre-import holidays here (with all warnings suppressed)
so that __warningregistry__ in that module is populated before pytest's
collection phase imports any test file. Python's warn() short-circuits on a
populated registry entry, so the warning never fires again — regardless of
pytest's filter stack.
"""

import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # suppress everything during this import
    import holidays as _h  # noqa: F401 — triggers the warn() once

    # Ensure registry is marked so re-imports in any forked context also skip it.
    _mod = sys.modules.get("holidays.deprecations.v1_incompatibility")
    if _mod is not None:
        getattr(_mod, "__warningregistry__", {}).update(
            {k: True for k in getattr(_mod, "__warningregistry__", {})}
        )

del _h, _mod
