"""Validation shared by configuration loading and direct Qdrant clients."""

import ipaddress
import re
from urllib.parse import urlsplit

from jarvis.profiles import ConfigurationError


def validate_qdrant_endpoint(url: str | None, host: str | None, port: int | None) -> None:
    if url is not None and not isinstance(url, str):
        raise ConfigurationError("QDRANT_URL must be a string")
    if url:
        if host is not None or port is not None:
            raise ConfigurationError("QDRANT_URL cannot be combined with QDRANT_HOST/QDRANT_PORT")
        try:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or "?" in url
                or "#" in url
                or "\\" in url
                or any(char.isspace() or ord(char) < 32 for char in url)
                or parsed.netloc.endswith(":")
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)
            ):
                raise ValueError
            _validate_host(parsed.hostname)
        except (ValueError, TypeError):
            raise ConfigurationError(
                "QDRANT_URL must be an HTTP(S) URL without credentials"
            ) from None
    else:
        _validate_host(host if host is not None else "localhost")
        if port is not None and (
            isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
        ):
            raise ConfigurationError("QDRANT_PORT must be an integer between 1 and 65535")


def _validate_host(host: str) -> None:
    if not isinstance(host, str):
        raise ConfigurationError("QDRANT_HOST must be a hostname or IP address")
    try:
        ipaddress.ip_address(host)
        return
    except ValueError:
        pass
    if not re.fullmatch(r"(?=.{1,253}\Z)[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.?", host) or any(
        not label or len(label) > 63 or label.startswith("-") or label.endswith("-")
        for label in host.rstrip(".").split(".")
    ):
        raise ConfigurationError("QDRANT_HOST must be a hostname or IP address")
