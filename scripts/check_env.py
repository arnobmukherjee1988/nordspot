"""
Startup environment check.
Reads .env.example, verifies every key is present in the running environment.
Usage: python scripts/check_env.py
"""

import os
import sys
from pathlib import Path


def check_env() -> None:
    env_example = Path(".env.example")
    if not env_example.exists():
        print("ERROR: .env.example not found. Run from project root.")
        sys.exit(1)

    required_keys = []
    for line in env_example.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            # Strip inline comments (e.g. SE3_LAT=59.33   # Stockholm)
            key = line.split("=")[0].strip()
            required_keys.append(key)

    print(f"\nNordSpot environment check ({len(required_keys)} variables)\n")
    print(f"{'Variable':<35} {'Status'}")
    print("-" * 50)

    missing = []
    for key in required_keys:
        value = os.environ.get(key)
        if value:
            print(f"{key:<35} OK  set")
        else:
            print(f"{key:<35} MISSING")
            missing.append(key)

    print()
    if missing:
        print(f"ERROR: {len(missing)} required variable(s) not set.")
        print("Copy .env.example to .env and fill in the missing values.")
        sys.exit(1)
    else:
        print("All environment variables set. Ready to start.")


if __name__ == "__main__":
    check_env()
