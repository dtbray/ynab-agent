"""OAuth credential manager for YNAB."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from ynab_agent.config import settings

AUTHORIZE_URL = "https://app.ynab.com/oauth/authorize"
TOKEN_URL = "https://app.ynab.com/oauth/token"
EXPIRY_SKEW_SECONDS = 300


@dataclass(frozen=True)
class AuthorizationRequest:
    """Values needed to complete a YNAB OAuth authorization request."""

    url: str
    state: str
    code_verifier: str


@dataclass(frozen=True)
class OAuthTokenStatus:
    """Non-secret status details for a stored OAuth token."""

    path: str
    exists: bool
    has_access_token: bool = False
    has_refresh_token: bool = False
    expires_at: int | None = None
    seconds_until_expiry: int | None = None
    refresh_recommended: bool = False


class OAuthTokenStore:
    """File-backed storage for YNAB OAuth token responses."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.ynab_oauth_token_path).expanduser()

    def load(self) -> dict[str, Any] | None:
        """Load a stored token response, if present."""
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, token: dict[str, Any]) -> None:
        """Persist a token response with a derived absolute expiration."""
        payload = dict(token)
        expires_in = int(payload.get("expires_in", 7200))
        payload["expires_at"] = int(time.time()) + expires_in

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.chmod(tmp_path, 0o600)
        tmp_path.replace(self.path)

    def has_valid_access_token(self) -> bool:
        """Return true when the current access token is not near expiry."""
        token = self.load()
        if not token:
            return False
        return int(token.get("expires_at", 0)) > int(time.time()) + EXPIRY_SKEW_SECONDS

    def status(self) -> OAuthTokenStatus:
        """Return non-secret observability details for the stored token."""
        token = self.load()
        if not token:
            return OAuthTokenStatus(path=str(self.path), exists=False)

        now = int(time.time())
        expires_at = _int_or_none(token.get("expires_at"))
        seconds_until_expiry = expires_at - now if expires_at is not None else None
        refresh_recommended = (
            seconds_until_expiry is None or seconds_until_expiry <= EXPIRY_SKEW_SECONDS
        )
        return OAuthTokenStatus(
            path=str(self.path),
            exists=True,
            has_access_token=bool(token.get("access_token")),
            has_refresh_token=bool(token.get("refresh_token")),
            expires_at=expires_at,
            seconds_until_expiry=seconds_until_expiry,
            refresh_recommended=refresh_recommended,
        )


class YnabOAuthManager:
    """Create YNAB OAuth requests and maintain access tokens."""

    def __init__(self, store: OAuthTokenStore | None = None):
        self.store = store or OAuthTokenStore()

    def authorization_request(self) -> AuthorizationRequest:
        """Build an authorization URL with state and PKCE."""
        self._require_client_id()
        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = _base64url_sha256(code_verifier)

        params = {
            "client_id": settings.ynab_oauth_client_id,
            "redirect_uri": settings.ynab_oauth_redirect_uri,
            "response_type": "code",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        if settings.ynab_oauth_scope:
            params["scope"] = settings.ynab_oauth_scope

        return AuthorizationRequest(
            url=f"{AUTHORIZE_URL}?{urlencode(params)}",
            state=state,
            code_verifier=code_verifier,
        )

    def exchange_code(self, code: str, code_verifier: str) -> dict[str, Any]:
        """Exchange an authorization code for access and refresh tokens."""
        data = {
            "client_id": self._require_client_id(),
            "client_secret": self._require_client_secret(),
            "redirect_uri": settings.ynab_oauth_redirect_uri,
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
        }
        token = self._post_token(data)
        self.store.save(token)
        return token

    def refresh(self) -> dict[str, Any]:
        """Refresh the stored access token and persist the rotated response."""
        current = self.store.load()
        if not current or not current.get("refresh_token"):
            raise ValueError("No YNAB OAuth refresh token is stored")

        data = {
            "client_id": self._require_client_id(),
            "client_secret": self._require_client_secret(),
            "grant_type": "refresh_token",
            "refresh_token": current["refresh_token"],
        }
        token = self._post_token(data)
        self.store.save(token)
        return token

    def access_token(self) -> str:
        """Return a valid access token, refreshing first when needed."""
        current = self.store.load()
        if current and self.store.has_valid_access_token():
            return str(current["access_token"])
        return str(self.refresh()["access_token"])

    def status(self) -> OAuthTokenStatus:
        """Return non-secret observability details for OAuth configuration."""
        return self.store.status()

    def _post_token(self, data: dict[str, str]) -> dict[str, Any]:
        response = requests.post(TOKEN_URL, data=data, timeout=30)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"YNAB OAuth token request failed: {response.text}") from exc
        token = response.json()
        if "access_token" not in token:
            raise RuntimeError("YNAB OAuth token response did not include access_token")
        return token

    @staticmethod
    def _require_client_id() -> str:
        if not settings.ynab_oauth_client_id:
            raise ValueError("YNAB_OAUTH_CLIENT_ID is required for OAuth")
        return settings.ynab_oauth_client_id

    @staticmethod
    def _require_client_secret() -> str:
        if not settings.ynab_oauth_client_secret:
            raise ValueError("YNAB_OAUTH_CLIENT_SECRET is required for OAuth")
        return settings.ynab_oauth_client_secret.get_secret_value()


def _base64url_sha256(value: str) -> str:
    digest = hashlib.sha256(value.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
