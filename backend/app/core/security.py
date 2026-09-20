"""Security utilities: URL sanitization, SSRF protection, and prompt injection defense."""

import ipaddress
import re
import socket
import urllib.parse


def is_safe_url(url: str) -> tuple[bool, str]:
    """
    Validates a URL against Server-Side Request Forgery (SSRF) risks.
    Blocks private IP spaces, cloud metadata endpoints, localhost, and non-http(s) schemes.
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"Invalid scheme: '{parsed.scheme}'. Only http and https are permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL does not contain a valid hostname"

    # Reject localhost variants
    if hostname.lower() in ("localhost", "127.0.0.1", "::1"):
        return False, "Localhost addresses are forbidden."

    # Check for IP address literals
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
            return False, f"Private or local IP addresses are forbidden: {hostname}"
    except ValueError:
        # Not an IP literal, it's a domain name. Resolve DNS to check resolved IP.
        try:
            resolved_ip_str = socket.gethostbyname(hostname)
            resolved_ip = ipaddress.ip_address(resolved_ip_str)
            if (
                resolved_ip.is_private
                or resolved_ip.is_loopback
                or resolved_ip.is_link_local
                or resolved_ip.is_reserved
            ):
                return (
                    False,
                    f"Host resolves to a private or restricted network address: {resolved_ip_str}",
                )
        except socket.gaierror:
            # If DNS resolution fails here, the fetcher will handle it, but it's not a private IP exploit
            pass

    return True, "URL is valid and safe"


def sanitize_input_text(text: str, max_chars: int = 50000) -> str:
    """
    Sanitizes raw article text:
    - Enforces length boundary
    - Strips non-printable control characters while preserving standard whitespace
    """
    if not text:
        return ""
    # Enforce maximum character boundary
    text = text[:max_chars]
    # Remove control characters except tab, newline, carriage return
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return sanitized.strip()


def wrap_untrusted_content(tag_name: str, content: str) -> str:
    """
    Encapsulates untrusted input in strict XML tags to prevent prompt injection.
    Escapes tag tokens within the content to avoid tag breakout.
    """
    escaped = content.replace(f"<{tag_name}>", f"&lt;{tag_name}&gt;").replace(
        f"</{tag_name}>", f"&lt;/{tag_name}&gt;"
    )
    return f"<{tag_name}>\n{escaped}\n</{tag_name}>"
