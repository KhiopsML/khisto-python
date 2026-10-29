# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging>=24.2"]
# ///

"""Emit the stable CPython minor versions matching the project's requires-python."""

import json
import os
import subprocess
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

with open("pyproject.toml", "rb") as stream:
    requirement = SpecifierSet(tomllib.load(stream)["project"]["requires-python"])

uv_output = subprocess.run(
    ["uv", "python", "list", "--all-versions", "--output-format", "json"],
    check=True,
    capture_output=True,
    text=True,
).stdout

versions = sorted(
    {
        Version(i["version"]).release[:2]
        for i in json.loads(uv_output)
        if i["implementation"] == "cpython"
        and i["variant"] == "default"
        and requirement.contains(i["version"], prereleases=False)
    }
)
if not versions:
    raise SystemExit("::error::No supported stable CPython versions found")

matrix = json.dumps([f"{major}.{minor}" for major, minor in versions])
print(matrix)
if output_file := os.environ.get("GITHUB_OUTPUT"):
    with Path(output_file).open("a", encoding="utf-8") as stream:
        stream.write(f"python-versions={matrix}\n")
