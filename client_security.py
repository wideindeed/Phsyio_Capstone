"""
client_security.py — Physio-Vision Client Security Layer
=========================================================
Responsibilities:
  1. SHA-256 password pre-hashing before any network call
     (server then bcrypt-hashes the hash → double-hashing for defense-in-depth)
  2. In-memory JWT session token storage
     (token is cleared the moment the process exits — never written to disk)
  3. Authorization header factory used by every authenticated API call

Import in auth.py and dashboard.py — never import from engine.py.
"""

import hashlib

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
# These are module-level strings so they act as a lightweight singleton.
# Nothing here is ever written to disk.
_session_token: str = ""
_session_user:  str = ""


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(plain_text: str) -> str:
    """
    SHA-256 pre-hash the password before it leaves the client.

    Why bother if the server also hashes?
    ➜ Defense-in-depth: even if TLS is misconfigured or an HTTP proxy is
      sitting between the Raspberry Pi and the client on the local network,
      the raw password string is never transmitted.
    ➜ The server receives a hex-digest string, not the actual password.
      bcrypt then hashes *that*, so the two layers are independent.
    """
    return hashlib.sha256(plain_text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

def store_session(token: str, username: str) -> None:
    """
    Called immediately after a successful /login response.
    Stores the JWT and the canonical username for the rest of the session.
    """
    global _session_token, _session_user
    _session_token = token
    _session_user  = username


def get_auth_headers() -> dict:
    """
    Returns the Authorization header dict ready to pass into requests.get/post.

    Usage:
        resp = requests.get(url, headers=get_auth_headers())

    Returns an empty dict (not None) when no session exists so callers
    never need to guard against None.
    """
    if not _session_token:
        return {}
    return {"Authorization": f"Bearer {_session_token}"}


def get_token() -> str:
    """Raw token string — use get_auth_headers() in most cases."""
    return _session_token


def get_username() -> str:
    """The username that owns the current session."""
    return _session_user


def is_authenticated() -> bool:
    """True when a valid-looking token is in memory."""
    return bool(_session_token)


def clear_session() -> None:
    """
    Wipe the in-memory session.
    Call on logout or when the server returns 401.
    """
    global _session_token, _session_user
    _session_token = ""
    _session_user  = ""
