# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Verifie les versions corrigees des clients utilises, pas les magasins CA."""

import argparse
import os
import re
import subprocess
import sys

MINIMUM_VERSIONS = {"uv": (0, 13, 4), "pip": (26, 2, 2)}


def main(argv: list[str] | None = None) -> int:
    """Bloque si le client utilise est trop ancien ou sa version inverifiable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", choices=MINIMUM_VERSIONS, default="uv")
    client = parser.parse_args(argv).client
    print(f"Runner: {os.environ.get('ImageOS')} {os.environ.get('ImageVersion')}")
    print(f"Python: {sys.version}; executable: {sys.executable}")
    print("Scope: patched package client version, not an audit of native CA stores.")
    command = (
        ["uv", "--version"]
        if client == "uv"
        else [sys.executable, "-m", "pip", "--version"]
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
    print(result.stdout.strip())
    match = re.match(rf"{client} (\d+)\.(\d+)(?:\.(\d+))?(?=\s|$)", result.stdout)
    if match is None:
        print(f"::error::{client}: cannot identify a stable version")
        return 1
    version = tuple(int(part or 0) for part in match.groups())
    minimum = MINIMUM_VERSIONS[client]
    if version < minimum:
        required = ".".join(map(str, minimum))
        print(f"::error::{client}: requires >= {required} for the eMudhra fix")
        return 1
    print(f"{client}: patched version check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
