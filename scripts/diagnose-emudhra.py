# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

# PEP 723 metadata lets uv run this standalone script with its own packaging dependency.
# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging>=24.2"]
# ///

"""Verifie les versions avec packaging, sans auditer les magasins CA."""

import argparse
import os
import subprocess
import sys

MINIMUM_VERSIONS = {"uv": "0.13.4", "pip": "26.2.2"}


def main(argv: list[str] | None = None) -> int:
    """Bloque si le client utilise est trop ancien ou sa version inverifiable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", choices=MINIMUM_VERSIONS, default="uv")
    client = parser.parse_args(argv).client
    print(f"Runner: {os.environ.get('ImageOS')} {os.environ.get('ImageVersion')}")
    print(f"Python: {sys.version}; executable: {sys.executable}")
    print("Scope: patched package client version, not an audit of native CA stores.")
    command = (
        ["uv", "self", "version", "--short"]
        if client == "uv"
        else [
            sys.executable,
            "-c",
            "from importlib.metadata import version; print(version('pip'))",
        ]
    )
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
    except subprocess.CalledProcessError as error:
        print(f"::error::{client}: version check failed: {error.stderr or error}")
        return 1
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"::error::{client}: version check inconclusive: {error}")
        return 1
    version_text = result.stdout.strip()
    print(f"{client}: {version_text}")
    try:
        if client == "pip":
            # Pip fournit ce parseur avant toute installation supplementaire.
            from pip._vendor.packaging.version import InvalidVersion, Version
        else:
            from packaging.version import InvalidVersion, Version
    except ImportError as error:
        print(f"::error::{client}: version parser unavailable: {error}")
        return 1
    try:
        version = Version(version_text)
    except InvalidVersion as error:
        print(f"::error::{client}: cannot identify a stable version: {error}")
        return 1
    if (
        version.is_prerelease
        or version.is_postrelease
        or version.local is not None
        or version.epoch != 0
        or len(version.release) not in (2, 3)
    ):
        print(f"::error::{client}: cannot identify a stable version")
        return 1
    minimum = MINIMUM_VERSIONS[client]
    if version < Version(minimum):
        print(f"::error::{client}: requires >= {minimum} for the eMudhra fix")
        return 1
    print(f"{client}: patched version check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
