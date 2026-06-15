import time
from collections import defaultdict
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Lightweight, dependency-free in-memory rate limiting middleware.
    Restricts the number of requests to specified routes per client IP.
    """
    def __init__(self, app, requests_limit: int = 10, window_seconds: int = 60):
        super().__init__(app)
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        # Maps client_ip -> list of request timestamps
        self.request_history = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Only rate limit the chat API route to protect Gemini API quota
        if request.url.path == "/api/chat" or request.url.path == "/api/chat/":
            client_ip = request.client.host if request.client else "unknown"
            current_time = time.time()
            
            # Filter out timestamps that are older than the window
            timestamps = self.request_history[client_ip]
            self.request_history[client_ip] = [t for t in timestamps if current_time - t < self.window_seconds]
            
            if len(self.request_history[client_ip]) >= self.requests_limit:
                # Rate limit triggered: return HTTP 429
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Too many requests. Please wait a moment before sending another message."
                    }
                )
            
            # Record current request timestamp
            self.request_history[client_ip].append(current_time)
            
        return await call_next(request)
