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
DISCOVERY = runpy.run_path(str(SCRIPT))


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.11"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    return tmp_path


def test_matrix_contains_sorted_unique_stable_cpython_versions(
    project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = (
        "cpython-3.15.1-linux-x86_64-gnu <download available>\n"
        "cpython-3.11.17-linux-x86_64-gnu <download available>\n"
        "cpython-3.15.0-linux-x86_64-gnu /installed/python\n"
        "cpython-3.12.15-linux-x86_64-gnu <download available>\n"
        "cpython-3.16.0a1-linux-x86_64-gnu <download available>\n"
        "cpython-3.15.1t-linux-x86_64-gnu <download available>\n"
        "pypy-3.11.17-linux-x86_64-gnu <download available>\n"
        "graalpy-3.13.0-linux-x86_64-gnu <download available>\n"
    )
    completed = subprocess.CompletedProcess("uv", 0, stdout=output)

    with patch("subprocess.run", return_value=completed) as run:
        assert DISCOVERY["main"]() == 0

    assert json.loads(capsys.readouterr().out) == ["3.11", "3.12", "3.15"]
    run.assert_called_once_with(
        ["uv", "python", "list", "--all-versions", ">=3.11"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        timeout=30,
    )


def test_project_requirement_is_used_and_versions_are_sorted_numerically(
    project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (project / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.9"\n', encoding="utf-8"
    )
    completed = subprocess.CompletedProcess(
        "uv",
        0,
        stdout="cpython-3.10.0-linux-x86_64-gnu\ncpython-3.9.0-linux-x86_64-gnu\n",
    )

    with patch("subprocess.run", return_value=completed) as run:
        assert DISCOVERY["main"]() == 0

    assert run.call_args.args[0][-1] == ">=3.9"
    assert json.loads(capsys.readouterr().out) == ["3.9", "3.10"]


def test_github_output_is_appended_without_overwriting_other_keys(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_file = project / "github-output"
    output_file.write_text("existing=value\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    completed = subprocess.CompletedProcess(
        "uv", 0, stdout="cpython-3.11.17-linux-x86_64-gnu\n"
    )

    with patch("subprocess.run", return_value=completed):
        assert DISCOVERY["main"]() == 0

    assert output_file.read_text(encoding="utf-8") == (
        'existing=value\npython-versions=["3.11"]\n'
    )


@pytest.mark.parametrize(
    "output",
    ["", "unexpected output", "cpython-3.16.0a1-linux-x86_64-gnu\n"],
)
def test_empty_or_unrecognizable_matrix_blocks_ci(
    project: Path,
    output: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = subprocess.CompletedProcess("uv", 0, stdout=output)

    with patch("subprocess.run", return_value=completed):
        assert DISCOVERY["main"]() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "::error::No supported stable CPython versions" in captured.err


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("uv missing"),
        PermissionError("cannot execute uv"),
        subprocess.TimeoutExpired("uv", 30),
        subprocess.CalledProcessError(1, "uv"),
    ],
)
def test_discovery_failure_blocks_ci(
    project: Path,
    error: Exception,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("subprocess.run", side_effect=error):
        assert DISCOVERY["main"]() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "::error::Cannot discover Python versions" in captured.err
