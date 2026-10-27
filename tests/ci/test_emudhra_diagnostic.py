# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Tests hors reseau du diagnostic des versions des clients de paquets."""

import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

DIAGNOSTIC = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "diagnose-emudhra.py")
)


@pytest.mark.parametrize(
    "client,output,exit_code",
    [
        ("uv", "0.13.3\n", 1),
        ("uv", "0.13.4\n", 0),
        ("uv", "0.13.5\n", 0),
        ("uv", "0.14.0\n", 0),
        ("uv", "1.0.0\n", 0),
        ("pip", "26.2.1\n", 1),
        ("pip", "26.2.2\n", 0),
        ("pip", "26.2\n", 1),
        ("pip", "27.0\n", 0),
        ("uv", "0.13.4rc1\n", 1),
        ("pip", "26.2.2.dev1\n", 1),
        ("pip", "26.2.2.post1\n", 1),
        ("uv", "0.13.4+local\n", 1),
        ("uv", "0.13.4.1\n", 1),
        ("uv", "0.13.4\nunexpected output\n", 1),
        ("uv", "0.13.4 (build)\n", 1),
        ("uv", "\uff10.\uff11\uff13.\uff14\n", 1),
        ("uv", "unknown\n", 1),
        ("uv", "pip 26.2.2\n", 1),
        ("pip", "", 1),
    ],
)
def test_only_identifiable_patched_clients_pass(
    client: str,
    output: str,
    exit_code: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = (
        ["uv", "self", "version", "--short"]
        if client == "uv"
        else [
            sys.executable,
            "-c",
            "from importlib.metadata import version; print(version('pip'))",
        ]
    )
    completed = subprocess.CompletedProcess(command, 0, stdout=output)

    with patch("subprocess.run", return_value=completed) as run:
        result = DIAGNOSTIC["main"](["--client", client])

    assert result == exit_code
    assert ("::error::" in capsys.readouterr().out) == (exit_code != 0)
    run.assert_called_once_with(
        command, check=True, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize("client", ["uv", "pip"])
@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("client missing"),
        PermissionError("cannot execute client"),
        subprocess.TimeoutExpired("client", 30),
        subprocess.CalledProcessError(1, "client", stderr="client failed"),
    ],
)
def test_missing_or_inconclusive_client_blocks_ci(
    client: str,
    error: Exception,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("subprocess.run", side_effect=error):
        assert DIAGNOSTIC["main"](["--client", client]) == 1

    assert f"::error::{client}:" in capsys.readouterr().out


def test_unknown_client_cannot_skip_the_gate() -> None:
    with patch("subprocess.run") as run, pytest.raises(SystemExit) as error:
        DIAGNOSTIC["main"](["--client", "unknown"])

    assert error.value.code == 2
    run.assert_not_called()


@pytest.mark.parametrize("version,exit_code", [("0.13.3", 1), ("0.13.4", 0)])
def test_cli_exit_status_enforces_the_uv_gate(version: str, exit_code: int) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "diagnose-emudhra.py"
    completed = subprocess.CompletedProcess(
        ["uv", "self", "version", "--short"], 0, stdout=f"{version}\n"
    )

    with (
        patch("sys.argv", [str(script)]),
        patch("subprocess.run", return_value=completed),
        pytest.raises(SystemExit) as error,
    ):
        runpy.run_path(str(script), run_name="__main__")

    assert error.value.code == exit_code
