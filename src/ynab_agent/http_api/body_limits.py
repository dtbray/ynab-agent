"""Bound raw planner request bodies before JSON parsing and validation."""

from __future__ import annotations

from collections import deque

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .dependencies import HttpApiRuntime


_BOUNDED_POST_PATHS = {
    "/planner/jobs",
    "/wealth/allocations/validate",
    "/wealth/scenarios/revisions",
    "/wealth/scenarios/comparisons",
    "/wealth/tax/strategies/jobs",
    "/wealth/social-security/optimize",
    "/wealth/housing/project",
}


class PlannerRequestBodyLimitMiddleware:
    """Reject oversized planning submissions, including streamed bodies."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in _BOUNDED_POST_PATHS
        ):
            await self.app(scope, receive, send)
            return

        application = scope["app"]
        runtime = getattr(application.state, "http_runtime", None)
        if not isinstance(runtime, HttpApiRuntime):
            await self.app(scope, receive, send)
            return
        maximum_bytes = runtime.api_settings.planner_max_request_body_bytes
        content_length = _content_length(scope)
        if content_length is not None and content_length > maximum_bytes:
            await _send_too_large(scope, receive, send)
            return

        buffered: deque[Message] = deque()
        received_bytes = 0
        more_body = True
        while more_body:
            message = await receive()
            buffered.append(message)
            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                continue
            received_bytes += len(message.get("body", b""))
            if received_bytes > maximum_bytes:
                await _send_too_large(scope, receive, send)
                return
            more_body = bool(message.get("more_body", False))

        async def replay() -> Message:
            if buffered:
                return buffered.popleft()
            return await receive()

        await self.app(scope, replay, send)


def _content_length(scope: Scope) -> int | None:
    for key, value in scope["headers"]:
        if key.lower() == b"content-length":
            try:
                return max(0, int(value))
            except ValueError:
                return None
    return None


async def _send_too_large(
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    if scope["path"] in {"/planner/jobs", "/wealth/tax/strategies/jobs"}:
        message = "planner job request body exceeds the configured limit"
    elif scope["path"] == "/wealth/social-security/optimize":
        message = "Social Security optimization request body exceeds the configured limit"
    else:
        message = "scenario request body exceeds the configured limit"
    response = JSONResponse(
        status_code=413,
        content={
            "detail": {
                "code": "request_too_large",
                "message": message,
            }
        },
    )
    await response(scope, receive, send)
