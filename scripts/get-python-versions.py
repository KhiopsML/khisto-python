# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging>=24.2"]
# ///

"""Use uv's structured version parts and packaging to emit stable CPython versions."""

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.version import Version


def main() -> int:
    """Discover available Python versions matching the project requirement."""
    with Path("pyproject.toml").open("rb") as stream:
        requirement = tomllib.load(stream)["project"]["requires-python"]
    try:
        result = subprocess.run(
            [
                "uv",
                "python",
                "list",
                "--all-versions",
                "--output-format",
                "json",
                requirement,
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"::error::Cannot discover Python versions: {error}", file=sys.stderr)
        return 1
    versions: set[tuple[int, int]] = set()
    try:
        installations = json.loads(result.stdout)
        if not isinstance(installations, list):
            raise TypeError("expected a list of Python installations")
        for installation in installations:
            if not isinstance(installation, dict) or not all(
                isinstance(installation.get(key), str)
                for key in ("implementation", "variant", "version")
            ):
                raise TypeError("invalid Python installation metadata")
            if (
                installation["implementation"] != "cpython"
                or installation["variant"] != "default"
            ):
                continue
            version = Version(installation["version"])
            if (
                version.is_prerelease
                or version.is_postrelease
                or version.local is not None
                or version.epoch != 0
            ):
                continue
            parts = installation.get("version_parts")
            if not isinstance(parts, dict) or not all(
                type(parts.get(key)) is int and parts[key] >= 0
                for key in ("major", "minor", "patch")
            ):
                raise TypeError("invalid Python version parts")
            if (parts["major"], parts["minor"], parts["patch"]) != version.release:
                raise ValueError("inconsistent Python version metadata")
            versions.add((parts["major"], parts["minor"]))
    except (TypeError, ValueError) as error:
        print(f"::error::Cannot parse Python versions: {error}", file=sys.stderr)
        return 1
    if not versions:
        print("::error::No supported stable CPython versions found", file=sys.stderr)
        return 1
    matrix = json.dumps([f"{major}.{minor}" for major, minor in sorted(versions)])
    print(matrix)
    if output_file := os.environ.get("GITHUB_OUTPUT"):
        with Path(output_file).open("a", encoding="utf-8") as stream:
            stream.write(f"python-versions={matrix}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
