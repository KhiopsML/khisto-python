# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Print the supported stable CPython versions and set the GitHub matrix output."""

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
            ["uv", "python", "list", "--all-versions", requirement],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"::error::Cannot discover Python versions: {error}", file=sys.stderr)
        return 1
    versions = sorted(
        {
            (int(major), int(minor))
            for major, minor in re.findall(
                r"cpython-(\d+)\.(\d+)(?=\.\d+-)", result.stdout
            )
        }
    )
    if not versions:
        print("::error::No supported stable CPython versions found", file=sys.stderr)
        return 1
    matrix = json.dumps([f"{major}.{minor}" for major, minor in versions])
    print(matrix)
    if output_file := os.environ.get("GITHUB_OUTPUT"):
        with Path(output_file).open("a", encoding="utf-8") as stream:
            stream.write(f"python-versions={matrix}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
