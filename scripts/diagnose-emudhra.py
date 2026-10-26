# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Diagnostique la confiance TLS du runner sans modifier ses certificats."""

import os
import socket
import ssl
import sys

# Test chains published at https://emudhra.we-have-a-problem.com/.
HOSTS = (
    "dv-g3b-emudhra.we-have-a-problem.com",
    "dv-g3b-aaa-emudhra.we-have-a-problem.com",
)


def rejects_certificate(host: str, context: ssl.SSLContext, client: str) -> bool:
    """Signale une chaine acceptee ou un diagnostic reseau non concluant."""
    try:
        with socket.create_connection((host, 443), timeout=20) as connection:
            with context.wrap_socket(connection, server_hostname=host):
                pass
    except ssl.SSLCertVerificationError as error:
        print(f"{client}: rejected {host}: {error}", flush=True)
        return True
    except OSError as error:
        print(f"::error::{client}: inconclusive TLS probe for {host}: {error}", flush=True)
        return False
    print(f"::error::{client}: runner trusts the affected chain at {host}", flush=True)
    return False


def main() -> int:
    """Controle le magasin Python et le bundle de pip sur l'image courante."""
    print(f"Runner: {os.environ.get('ImageOS')} {os.environ.get('ImageVersion')}")
    print(f"Python: {sys.version}")
    print(f"TLS: {ssl.OPENSSL_VERSION}; CA paths: {ssl.get_default_verify_paths()}")
    contexts = {"Python default": ssl.create_default_context()}
    try:
        import pip
        from pip._vendor import certifi
    except ModuleNotFoundError as error:
        if error.name != "pip":
            raise
        print("pip: not installed in this interpreter")
    else:
        print(f"pip: {pip.__version__}; vendored certifi: {certifi.__version__}")
        contexts["pip CA bundle"] = ssl.create_default_context(cafile=certifi.where())
    results = []
    for client, context in contexts.items():
        try:
            with socket.create_connection(("pypi.org", 443), timeout=20) as connection:
                with context.wrap_socket(connection, server_hostname="pypi.org"):
                    pass
        except OSError as error:
            print(f"::error::{client}: inconclusive: cannot verify pypi.org: {error}")
            results.append(False)
            continue
        results.extend(rejects_certificate(host, context, client) for host in HOSTS)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
