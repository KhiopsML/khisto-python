# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Read uv's JSON catalog and emit the supported stable CPython matrix."""

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path


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
            match = re.fullmatch(r"([0-9]+)\.([0-9]+)\.[0-9]+", installation["version"])
            if match:
                versions.add((int(match[1]), int(match[2])))
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
