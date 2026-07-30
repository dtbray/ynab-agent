import json
import time
from urllib.parse import parse_qs, urlparse

from ynab_agent.config import settings
from ynab_agent.credentials.oauth import EXPIRY_SKEW_SECONDS, OAuthTokenStore, YnabOAuthManager


def test_authorization_request_uses_pkce_and_state(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ynab_oauth_client_id", "client-id")
    monkeypatch.setattr(settings, "ynab_oauth_redirect_uri", "http://localhost:8765/callback")
    monkeypatch.setattr(settings, "ynab_oauth_scope", "read-only")

    request = YnabOAuthManager(OAuthTokenStore(tmp_path / "token.json")).authorization_request()
    params = parse_qs(urlparse(request.url).query)

    assert params["client_id"] == ["client-id"]
    assert params["redirect_uri"] == ["http://localhost:8765/callback"]
    assert params["response_type"] == ["code"]
    assert params["state"] == [request.state]
    assert params["code_challenge_method"] == ["S256"]
    assert params["scope"] == ["read-only"]
    assert len(request.code_verifier) >= 43
    assert params["code_challenge"][0] != request.code_verifier


def test_token_store_saves_expiry_and_private_permissions(tmp_path):
    path = tmp_path / "token.json"
    store = OAuthTokenStore(path)

    store.save({"access_token": "access", "refresh_token": "refresh", "expires_in": 600})
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert stored["access_token"] == "access"
    assert stored["refresh_token"] == "refresh"
    assert stored["expires_at"] > int(time.time())
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_access_token_refreshes_expired_token(monkeypatch, tmp_path):
    path = tmp_path / "token.json"
    store = OAuthTokenStore(path)
    store.save({"access_token": "old", "refresh_token": "refresh", "expires_in": -10})

    monkeypatch.setattr(settings, "ynab_oauth_client_id", "client-id")
    monkeypatch.setattr(settings, "ynab_oauth_client_secret", _Secret("client-secret"))

    def fake_post(url, data, timeout):
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "refresh"
        return _Response({"access_token": "new", "refresh_token": "next", "expires_in": 600})

    monkeypatch.setattr("ynab_agent.credentials.oauth.requests.post", fake_post)

    token = YnabOAuthManager(store).access_token()

    assert token == "new"
    assert store.load()["refresh_token"] == "next"


def test_access_token_refreshes_token_before_expiry_skew(monkeypatch, tmp_path):
    path = tmp_path / "token.json"
    store = OAuthTokenStore(path)
    store.save(
        {
            "access_token": "old",
            "refresh_token": "refresh",
            "expires_in": EXPIRY_SKEW_SECONDS - 1,
        }
    )

    monkeypatch.setattr(settings, "ynab_oauth_client_id", "client-id")
    monkeypatch.setattr(settings, "ynab_oauth_client_secret", _Secret("client-secret"))

    def fake_post(url, data, timeout):
        assert data["grant_type"] == "refresh_token"
        return _Response({"access_token": "new", "refresh_token": "next", "expires_in": 600})

    monkeypatch.setattr("ynab_agent.credentials.oauth.requests.post", fake_post)

    assert YnabOAuthManager(store).access_token() == "new"


def test_token_status_exposes_non_secret_observability(tmp_path):
    path = tmp_path / "token.json"
    store = OAuthTokenStore(path)

    missing = store.status()
    assert missing.exists is False
    assert missing.path == str(path)

    store.save({"access_token": "access", "refresh_token": "refresh", "expires_in": 600})
    status = store.status()

    assert status.exists is True
    assert status.has_access_token is True
    assert status.has_refresh_token is True
    assert status.expires_at is not None
    assert status.seconds_until_expiry is not None
    assert status.seconds_until_expiry > 0
    assert status.refresh_recommended is False


class _Secret:
    def __init__(self, value):
        self.value = value

    def get_secret_value(self):
        return self.value


class _Response:
    def __init__(self, payload):
        self.payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload
