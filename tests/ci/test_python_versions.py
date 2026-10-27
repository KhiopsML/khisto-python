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


def installation(
    version: str, implementation: str = "cpython", variant: str = "default"
) -> dict[str, str]:
    return {
        "version": version,
        "implementation": implementation,
        "variant": variant,
    }


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
    output = json.dumps(
        [
            installation("3.15.1"),
            installation("3.11.17"),
            installation("3.15.0"),
            installation("3.12.15"),
            installation("3.16.0a1"),
            installation("3.15.1", variant="freethreaded"),
            installation("3.15.1", variant="debug"),
            installation("3.11.17", implementation="pypy"),
            installation("3.13.0", implementation="graalpy"),
        ]
    )
    completed = subprocess.CompletedProcess("uv", 0, stdout=output)

    with patch("subprocess.run", return_value=completed) as run:
        assert DISCOVERY["main"]() == 0

    assert json.loads(capsys.readouterr().out) == ["3.11", "3.12", "3.15"]
    run.assert_called_once_with(
        [
            "uv",
            "python",
            "list",
            "--all-versions",
            "--output-format",
            "json",
            ">=3.11",
        ],
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
        stdout=json.dumps([installation("3.10.0"), installation("3.9.0")]),
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
        "uv", 0, stdout=json.dumps([installation("3.11.17")])
    )

    with patch("subprocess.run", return_value=completed):
        assert DISCOVERY["main"]() == 0

    assert output_file.read_text(encoding="utf-8") == (
        'existing=value\npython-versions=["3.11"]\n'
    )


@pytest.mark.parametrize(
    "installations",
    [
        [],
        [installation("3.16.0a1")],
        [installation("3.16.0rc1")],
        [installation("3.16.0.dev1")],
        [installation("3.15.1", variant="freethreaded")],
        [installation("3.15.1", implementation="pypy")],
        [installation("\uff13.\uff11\uff15.\uff11")],
    ],
)
def test_matrix_without_stable_cpython_blocks_ci(
    project: Path,
    installations: list[dict[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = subprocess.CompletedProcess("uv", 0, stdout=json.dumps(installations))

    with patch("subprocess.run", return_value=completed):
        assert DISCOVERY["main"]() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "::error::No supported stable CPython versions" in captured.err


@pytest.mark.parametrize(
    "output",
    [
        "",
        "unexpected output",
        "{}",
        "null",
        "[null]",
        '["cpython-3.15.1-linux-x86_64-gnu"]',
        '[{"version": "3.15.1"}]',
        json.dumps([{**installation("3.15.1"), "version": None}]),
        json.dumps([{**installation("3.15.1"), "implementation": False}]),
        json.dumps([{**installation("3.15.1"), "variant": 1}]),
        json.dumps([installation("3.15.1"), {}]),
    ],
)
def test_malformed_discovery_output_blocks_ci_without_emitting_a_matrix(
    project: Path,
    output: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_file = project / "github-output"
    output_file.write_text("existing=value\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    completed = subprocess.CompletedProcess("uv", 0, stdout=output)

    with patch("subprocess.run", return_value=completed):
        assert DISCOVERY["main"]() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "::error::Cannot parse Python versions" in captured.err
    assert output_file.read_text(encoding="utf-8") == "existing=value\n"


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
