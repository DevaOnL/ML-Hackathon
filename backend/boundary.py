"""Small public HTTP boundary: size limits, safe errors, IDs and security headers."""
from __future__ import annotations

import asyncio
import time
import uuid
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

from .service import log_event

MAX_REQUEST_BYTES = 1024
SECURITY_HEADERS = {
    b"cache-control": b"no-store, no-transform",
    b"x-content-type-options": b"nosniff",
    b"referrer-policy": b"no-referrer",
    b"x-frame-options": b"DENY",
    b"permissions-policy": b"geolocation=(), camera=(), microphone=()",
    b"content-security-policy": b"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https://*.tile.openstreetmap.org; connect-src 'self' https: http:; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'",
}


def _origin_key(value: str):
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.path or parsed.query or parsed.fragment
                or port == 0 or any(character.isspace() for character in value)):
            return None
        return parsed.scheme.lower(), parsed.hostname.lower(), port if port is not None else (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None


class PublicBoundary:
    def __init__(self, app, *, allowed_origins=()):
        self.app = app
        self.allowed_origins = frozenset(key for value in allowed_origins if (key := _origin_key(value)) is not None)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        status_code = 500
        response_started = False
        started = time.perf_counter()
        path = scope.get("path", "")
        route = path if path in {"/health", "/trips", "/match"} else "static_or_unknown"

        async def safe_send(message):
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = [(key, value) for key, value in message.get("headers", []) if key.lower() not in SECURITY_HEADERS and key.lower() != b"x-request-id"]
                message = dict(message, headers=headers + list(SECURITY_HEADERS.items()) + [(b"x-request-id", request_id.encode("ascii"))])
            await send(message)

        async def error(status, code, message):
            await JSONResponse({"detail": {"code": code, "message": message, "request_id": request_id}}, status_code=status)(scope, receive, safe_send)

        try:
            next_receive = receive
            if scope.get("method") == "POST" and path.rstrip("/") == "/match":
                headers = {key.lower(): value for key, value in scope.get("headers", [])}
                origins = [value for key, value in scope.get("headers", []) if key.lower() == b"origin"]
                if origins:
                    origin = _origin_key(origins[0].decode("latin-1")) if len(origins) == 1 else None
                    host = headers.get(b"host", b"").decode("latin-1")
                    # Trust only the ASGI scheme. Uvicorn may set it from a trusted
                    # loopback proxy; arbitrary forwarded headers are not read here.
                    same_origin = _origin_key(f"{scope.get('scheme', 'http')}://{host}")
                    if origin is None or (origin != same_origin and origin not in self.allowed_origins):
                        await error(403, "origin_not_allowed", "This origin is not allowed to invoke inference.")
                        return
                if headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower() != b"application/json":
                    await error(415, "unsupported_media_type", "POST /match accepts application/json containing only trip_id.")
                    return
                if headers.get(b"content-encoding", b"identity").lower() != b"identity":
                    await error(415, "unsupported_content_encoding", "Compressed request bodies are not accepted.")
                    return
                content_length = headers.get(b"content-length")
                if content_length is not None:
                    try:
                        length = int(content_length)
                    except ValueError:
                        await error(400, "invalid_content_length", "The request Content-Length is invalid.")
                        return
                    if length < 0:
                        await error(400, "invalid_content_length", "The request Content-Length is invalid.")
                        return
                    if length > MAX_REQUEST_BYTES:
                        await error(413, "request_too_large", "POST /match accepts at most 1024 bytes.")
                        return
                chunks = []
                size = 0
                deadline = asyncio.get_running_loop().time() + 10.0
                while True:
                    try:
                        message = await asyncio.wait_for(receive(), timeout=max(0.001, deadline - asyncio.get_running_loop().time()))
                    except asyncio.TimeoutError:
                        await error(408, "request_timeout", "The JSON request body did not arrive in time.")
                        return
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > MAX_REQUEST_BYTES:
                        await error(413, "request_too_large", "POST /match accepts at most 1024 bytes.")
                        return
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
                body = b"".join(chunks)
                delivered = False

                async def replay():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return await receive()

                next_receive = replay
            await self.app(scope, next_receive, safe_send)
        except Exception as exc:
            log_event("http_internal_error", request_id=request_id, endpoint=route, error_type=type(exc).__name__)
            if response_started:
                raise
            await error(500, "internal_error", "The server could not complete this request. Consult the server log using this request ID.")
        finally:
            log_event("http_request", request_id=request_id, endpoint=route,
                      method=scope.get("method", ""), status_code=status_code,
                      elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3))
