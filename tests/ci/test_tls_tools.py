# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Tests des seuils de securite du bootstrap CI."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_tls_tools",
    Path(__file__).resolve().parents[2] / "scripts" / "check-tls-tools.py",
)
assert SPEC is not None and SPEC.loader is not None
TLS_TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TLS_TOOLS)


@pytest.mark.parametrize("name,minimum", TLS_TOOLS.MINIMUM_VERSIONS.items())
def test_accepts_patched_minimum(name: str, minimum: str) -> None:
    TLS_TOOLS.check_version(name, minimum)


@pytest.mark.parametrize(
    "name,installed",
    [
        ("uv", "0.11.32"),
        ("uv", "0.13.3"),
        ("pip", "25.3"),
        ("pip", "26.2.1"),
        ("certifi", "2026.7.22"),
        ("certifi", "2026.10.18"),
        ("sigstore", "4.5.0"),
    ],
)
def test_rejects_affected_versions(name: str, installed: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        TLS_TOOLS.check_version(name, installed)


@pytest.mark.parametrize("installed", ["26.2.2rc1", "26.2.2.dev0", "unknown"])
def test_rejects_unverified_versions(installed: str) -> None:
    with pytest.raises(ValueError, match="unsupported version"):
        TLS_TOOLS.check_version("pip", installed)


def test_rejects_compromised_root(monkeypatch: pytest.MonkeyPatch) -> None:
    class Context:
        def get_ca_certs(self) -> list[dict]:
            return [{"subject": ((("commonName", "emSign ECC Root CA - G3"),),)}]

    monkeypatch.setattr(TLS_TOOLS.ssl, "create_default_context", lambda **_: Context())
    with pytest.raises(ValueError, match="still trusts"):
        TLS_TOOLS.check_certificates()
