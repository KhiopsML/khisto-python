# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Offline tests of the Python CI matrix discovery."""

import json
import runpy
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "get-python-versions.py"


def installation(
    version: str, implementation: str = "cpython", variant: str = "default"
) -> dict[str, str]:
    return {"version": version, "implementation": implementation, "variant": variant}


def discover(installations: list[dict[str, str]]) -> None:
    completed = subprocess.CompletedProcess("uv", 0, stdout=json.dumps(installations))
    with patch("subprocess.run", return_value=completed):
        runpy.run_path(str(SCRIPT))


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.11"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    return tmp_path


def test_matrix_contains_sorted_unique_stable_cpython_versions(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    discover(
        [
            installation("3.15.1"),
            installation("3.11.17"),
            installation("3.15.0"),
            installation("3.12.15"),
            installation("3.10.12"),
            installation("3.16.0a1"),
            installation("3.15.1", variant="freethreaded"),
            installation("3.11.17", implementation="pypy"),
        ]
    )

    assert json.loads(capsys.readouterr().out) == ["3.11", "3.12", "3.15"]


def test_full_specifier_is_applied_and_sorted_numerically(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.9,<3.11"\n', encoding="utf-8"
    )

    discover([installation("3.11.0"), installation("3.10.0"), installation("3.9.0")])

    assert json.loads(capsys.readouterr().out) == ["3.9", "3.10"]


def test_github_output_is_appended(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_file = project / "github-output"
    output_file.write_text("existing=value\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    discover([installation("3.11.17")])

    assert output_file.read_text(encoding="utf-8") == (
        'existing=value\npython-versions=["3.11"]\n'
    )


@pytest.mark.parametrize(
    "installations",
    [[], [installation("3.16.0a1")], [installation("3.15.1", variant="debug")]],
)
def test_no_stable_cpython_blocks_ci(
    project: Path, installations: list[dict[str, str]]
) -> None:
    with pytest.raises(SystemExit, match="No supported stable CPython"):
        discover(installations)


def test_uv_failure_blocks_ci(project: Path) -> None:
    with (
        patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "uv")),
        pytest.raises(subprocess.CalledProcessError),
    ):
        runpy.run_path(str(SCRIPT))
