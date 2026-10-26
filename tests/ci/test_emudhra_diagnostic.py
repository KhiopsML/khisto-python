# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Tests hors reseau du diagnostic des certificats du runner."""

import runpy
import ssl
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

DIAGNOSTIC = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "diagnose-emudhra.py")
)


@pytest.mark.parametrize(
    "error,expected,message",
    [
        (None, False, "runner trusts"),
        (ssl.SSLCertVerificationError("revoked"), True, "rejected"),
        (TimeoutError("timed out"), False, "inconclusive"),
        (OSError("DNS unavailable"), False, "inconclusive"),
    ],
)
def test_probe_reports_actual_tls_result(
    error: OSError | None,
    expected: bool,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = MagicMock(spec=ssl.SSLContext)
    context.wrap_socket.side_effect = error

    with patch("socket.create_connection"):
        result = DIAGNOSTIC["rejects_certificate"](
            "example.test", context, "test client"
        )

    assert result is expected
    assert message in capsys.readouterr().out
    assert context.wrap_socket.call_args.kwargs["server_hostname"] == "example.test"


def test_connection_failure_is_not_certificate_rejection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("socket.create_connection", side_effect=TimeoutError("timed out")):
        result = DIAGNOSTIC["rejects_certificate"](
            "example.test", MagicMock(spec=ssl.SSLContext), "test client"
        )

    assert result is False
    assert "::error::test client: inconclusive" in capsys.readouterr().out


@pytest.mark.parametrize("rejected,exit_code", [(True, 0), (False, 1)])
def test_exit_code_matches_probe_results(rejected: bool, exit_code: int) -> None:
    with (
        patch("socket.create_connection"),
        patch("ssl.create_default_context"),
        patch.dict(
            DIAGNOSTIC["main"].__globals__,
            {"rejects_certificate": MagicMock(return_value=rejected)},
        ),
    ):
        assert DIAGNOSTIC["main"]() == exit_code


def test_empty_trust_store_is_inconclusive(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "socket.create_connection",
        side_effect=ssl.SSLCertVerificationError("no trusted CA certificates"),
    ):
        assert DIAGNOSTIC["main"]() == 1

    assert "inconclusive: cannot verify pypi.org" in capsys.readouterr().out
