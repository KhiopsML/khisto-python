# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Verifie les clients TLS utilises apres l'incident emSign."""

import argparse
import hashlib
import importlib.metadata
import re
import ssl
import subprocess
from pathlib import Path

MINIMUM_VERSIONS = {
    "uv": "0.13.4",
    "pip": "26.2.2",
    "certifi": "2026.10.19",
    "sigstore": "4.5.1",
}


def check_version(name: str, installed: str) -> None:
    """Refuse une version ancienne ou dont la stabilite est indeterminee."""
    minimum = MINIMUM_VERSIONS[name]
    if not re.fullmatch(r"\d+(?:\.\d+)+", installed):
        raise ValueError(f"{name}: unsupported version {installed!r}")
    if tuple(map(int, installed.split("."))) < tuple(map(int, minimum.split("."))):
        raise ValueError(f"{name} {installed} is unsafe; require >= {minimum}")
    print(f"{name}: {installed} (minimum {minimum})", flush=True)


def check_context(context: ssl.SSLContext) -> None:
    """Refuse un magasin vide ou contenant la racine compromise."""
    certificates = context.get_ca_certs()
    if not certificates:
        raise ValueError("TLS trust store is empty")
    for certificate in certificates:
        for rdn in certificate["subject"]:
            if ("commonName", "emSign ECC Root CA - G3") in rdn:
                raise ValueError("TLS trust store still trusts emSign ECC Root CA - G3")
    print(f"Checked {len(certificates)} loaded CA certificates")


def check_certificates() -> None:
    """Verifie le bundle effectif de certifi et journalise son empreinte."""
    import certifi

    bundle = Path(certifi.where())
    check_context(ssl.create_default_context(cafile=str(bundle)))
    print(f"certifi bundle: {bundle}")
    print(f"certifi SHA256: {hashlib.sha256(bundle.read_bytes()).hexdigest()}")


def main() -> None:
    """Controle les outils effectivement utilises par le job appelant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+", choices=[*MINIMUM_VERSIONS, "default"])
    arguments = parser.parse_args()
    for name in arguments.packages:
        if name == "default":
            check_context(ssl.create_default_context())
            print(f"Default CA paths: {ssl.get_default_verify_paths()}")
            continue
        if name == "uv":
            installed = subprocess.check_output(
                ["uv", "--version"], text=True
            ).split()[1]
        else:
            installed = importlib.metadata.version(name)
        check_version(name, installed)
        if name == "certifi":
            check_certificates()


if __name__ == "__main__":
    main()
