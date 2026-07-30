"""OAuth Typer commands."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Annotated

import typer

from ynab_agent.cli_support import console, print_json, render_rows
from ynab_agent.config import settings


oauth_app = typer.Typer(help="Manage YNAB OAuth authorization")


def _format_epoch(epoch_seconds: int | None) -> str | None:
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat()


@oauth_app.command("url")
def oauth_url(
    save_state: Annotated[
        bool,
        typer.Option(
            "--save-state/--no-save-state",
            help="Save PKCE state locally for the exchange command",
        ),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of text"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Only print the authorization URL",
        ),
    ] = False,
) -> None:
    """Print a YNAB authorization URL using Authorization Code + PKCE."""
    from ynab_agent.credentials.oauth import YnabOAuthManager

    request = YnabOAuthManager().authorization_request()
    output: dict[str, object] = {
        "url": request.url,
        "state": request.state,
        "code_verifier": request.code_verifier,
        "redirect_uri": settings.ynab_oauth_redirect_uri,
        "state_path": None,
    }

    if save_state:
        state_path = Path(".ynab-oauth-pkce.json")
        output["state_path"] = str(state_path)
        state_path.write_text(
            json.dumps(
                {
                    "state": request.state,
                    "code_verifier": request.code_verifier,
                    "redirect_uri": settings.ynab_oauth_redirect_uri,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        state_path.chmod(0o600)

    if json_output:
        print_json(output)
        return

    console.print(request.url)
    if save_state and not quiet:
        console.print(f"Saved OAuth state to {output['state_path']}")


@oauth_app.command("exchange")
def oauth_exchange(
    code: Annotated[
        str,
        typer.Argument(help="Authorization code from the redirect URL"),
    ],
    state: Annotated[
        str | None,
        typer.Option(
            "--state",
            help="State returned in the redirect URL",
        ),
    ] = None,
    code_verifier: Annotated[
        str | None,
        typer.Option(
            "--code-verifier",
            help="PKCE code verifier if not using saved state",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of text"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress success output",
        ),
    ] = False,
) -> None:
    """Exchange a returned authorization code for stored OAuth tokens."""
    from ynab_agent.credentials.oauth import YnabOAuthManager

    resolved_verifier = code_verifier
    if not resolved_verifier:
        state_path = Path(".ynab-oauth-pkce.json")
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if state and saved.get("state") != state:
            raise typer.BadParameter("Returned OAuth state does not match saved state")
        resolved_verifier = str(saved["code_verifier"])

    token = YnabOAuthManager().exchange_code(code, resolved_verifier)
    output = {
        "token_path": settings.ynab_oauth_token_path,
        "expires_in": token.get("expires_in"),
    }
    if json_output:
        print_json(output)
        return
    if quiet:
        return
    console.print(
        f"Stored OAuth token at {settings.ynab_oauth_token_path}; "
        f"expires in {token.get('expires_in', 'unknown')} seconds"
    )


@oauth_app.command("refresh")
def oauth_refresh(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of text"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress success output",
        ),
    ] = False,
) -> None:
    """Refresh and persist the stored YNAB OAuth token."""
    from ynab_agent.credentials.oauth import YnabOAuthManager

    token = YnabOAuthManager().refresh()
    output = {
        "token_path": settings.ynab_oauth_token_path,
        "expires_in": token.get("expires_in"),
    }
    if json_output:
        print_json(output)
        return
    if quiet:
        return
    console.print(f"Refreshed OAuth token; expires in {token.get('expires_in', 'unknown')} seconds")


@oauth_app.command("status")
def oauth_status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Exit with status only; print nothing",
        ),
    ] = False,
) -> None:
    """Show non-secret OAuth token status."""
    from ynab_agent.credentials.oauth import YnabOAuthManager

    status = YnabOAuthManager().status()
    data: dict[str, object] = {
        "token_path": status.path,
        "exists": status.exists,
        "has_access_token": status.has_access_token,
        "has_refresh_token": status.has_refresh_token,
        "expires_at": status.expires_at,
        "expires_at_iso": _format_epoch(status.expires_at),
        "seconds_until_expiry": status.seconds_until_expiry,
        "refresh_recommended": status.refresh_recommended,
    }
    is_usable = status.exists and status.has_access_token and status.has_refresh_token

    if quiet:
        raise typer.Exit(code=0 if is_usable else 1)
    if json_output:
        print_json(data)
        raise typer.Exit(code=0 if is_usable else 1)
    if not status.exists:
        console.print(f"No OAuth token found at {status.path}")
        raise typer.Exit(code=1)

    render_rows(
        [data],
        [
            ("token_path", "Token Path"),
            ("has_access_token", "Access Token"),
            ("has_refresh_token", "Refresh Token"),
            ("expires_at_iso", "Expires At"),
            ("seconds_until_expiry", "Seconds Left"),
            ("refresh_recommended", "Refresh Recommended"),
        ],
    )
