from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request
from app.core.security import decode_token


def get_rate_limit_key(request: Request) -> str:
    """
    Rate-limit per authenticated user when possible. Falls back to IP
    for unauthenticated routes (e.g. /auth/login, /auth/register).
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        payload = decode_token(auth_header[7:])
        if payload and payload.get("sub"):
            return f"user:{payload['sub']}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=get_rate_limit_key)