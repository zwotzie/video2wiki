"""Ensure SSL_CERT_FILE is set for internal CAs."""

import os
from pathlib import Path

_CANDIDATES = [
    "/opt/homebrew/etc/openssl@1.1/cert.pem",
    "/opt/homebrew/etc/openssl@3/cert.pem",
    "/etc/ssl/cert.pem",
    "/etc/pki/tls/cert.pem",
]


def ensure_ssl_cert() -> None:
    if "SSL_CERT_FILE" in os.environ:
        return
    for path in _CANDIDATES:
        if Path(path).exists():
            os.environ["SSL_CERT_FILE"] = path
            break


ensure_ssl_cert()
