"""
Generic FastAPI service for gym environments.

This service is completely reusable and environment-agnostic.
It only handles:
- HTTP routing
- Session ID validation
- Request/response encoding/decoding
- Forwarding to handler

All business logic is delegated to the handler.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from .handler import BaseGymHandler
from .multipart_codec import encode_multipart, parse_data_field, read_images, decode_multipart

LOGGER = logging.getLogger(__name__)

# ----------------------------
# Config (env-driven)
# ----------------------------
API_KEY = os.getenv("GYM_API_KEY", "")  # Empty => no auth
MAX_INFLIGHT = int(os.getenv("GYM_MAX_INFLIGHT", "0"))  # 0 => unlimited
ADMIT_TIMEOUT = float(os.getenv("GYM_ADMIT_TIMEOUT", "5.0"))

IMAGE_FORMAT = os.getenv("GYM_IMAGE_FORMAT", "PNG")
IMAGE_MIME = os.getenv("GYM_IMAGE_MIME", "image/png")

# Global concurrency limiter (optional)
_sem = asyncio.Semaphore(MAX_INFLIGHT) if MAX_INFLIGHT > 0 else None

# Request log file
REQUEST_LOG_PATH = Path(os.getenv("GYM_REQUEST_LOG", "request_log.jsonl"))


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Logs every request to a JSONL file for debugging."""

    async def dispatch(self, request: Request, call_next):
        t0 = time.time()

        # Read form data for POST requests (need to cache body for downstream)
        body_data = None
        if request.method == "POST":
            # We parse the 'data' field from the form later in the endpoint,
            # so here just record the raw query info.
            pass

        response = await call_next(request)
        elapsed = time.time() - t0

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
            "client": request.client.host if request.client else "unknown",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "elapsed_s": round(elapsed, 3),
        }

        # Add query params if present
        if request.query_params:
            entry["query"] = dict(request.query_params)

        try:
            with open(REQUEST_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        return response


def _auth(request: Request) -> None:
    """
    Optional API key authentication.

    Accepts:
      - Query param: ?token=...
      - Header: X-API-Key: ...
    """
    if not API_KEY:
        return
    token = request.query_params.get("token") or request.headers.get("x-api-key")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")


def build_gym_service(handler: BaseGymHandler) -> FastAPI:
    """
    Build a generic gym environment service.

    This function creates a FastAPI app that routes requests to the provided handler.
    The service is completely reusable - just provide a different handler for
    different environments.

    Args:
        handler: Handler instance that implements environment logic

    Returns:
        FastAPI application ready to serve

    Usage:
        >>> handler = MyGymHandler()
        >>> app = build_gym_service(handler)
        >>> # Run with: uvicorn mymodule:app --host 0.0.0.0 --port 8000
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await handler.aclose()

    app = FastAPI(
        title="Gym Environment Service",
        description="Generic HTTP service for remote gym environments",
        lifespan=lifespan,
    )
    app.state.handler = handler
    app.add_middleware(RequestLogMiddleware)

    def _log_request_detail(path: str, data_dict: dict, extra: dict = None):
        """Log detailed request info (form data) to the JSONL file."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "detail",
            "path": path,
            "data": data_dict,
        }
        if extra:
            entry.update(extra)
        try:
            with open(REQUEST_LOG_PATH, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "ok": True,
            "service": "gym-env-service",
            "max_inflight": MAX_INFLIGHT if MAX_INFLIGHT > 0 else "unlimited",
        }

    @app.get("/sessions")
    async def sessions(request: Request):
        """
        Get statistics about active sessions.

        Returns:
            JSON with session statistics:
            - num_sessions: Current number of active sessions
            - max_sessions: Maximum allowed sessions
            - sessions: List of session details (session_id, idle_time, etc.)

        Requires authentication if API_KEY is set.
        """
        _auth(request)

        stats = handler.get_session_stats()
        return stats

    @app.post("/connect")
    async def connect(
        request: Request,
        data: Optional[str] = Form(default=None),
        images: Optional[List[UploadFile]] = File(default=None),
    ):
        """
        Create a new session.

        Request:
            Content-Type: multipart/form-data
            - data: JSON string with env_config and optional seed
                    {"env_config": {...}, "seed": 42}  (seed is optional)

        Response:
            Content-Type: multipart/mixed
            - data: {"session_id": "..."}
                    If seed provided: also {"obs": "...", "info": {...}}
            - images: (if seed provided and env returns images)

        If seed is provided, the server will create the session AND perform
        initial reset in one round-trip, returning both session_id and reset result.
        """
        _auth(request)

        acquired = False
        if _sem is not None:
            try:
                await asyncio.wait_for(_sem.acquire(), timeout=ADMIT_TIMEOUT)
                acquired = True
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail="server busy")

        try:
            data_dict = parse_data_field(data)
            env_config = data_dict.get("env_config", {})
            seed = data_dict.get("seed")  # Optional
            _log_request_detail("/connect", data_dict)

            result = await handler.connect(env_config, seed=seed)

            boundary, body = encode_multipart(
                result.data,
                result.images,
                image_format=IMAGE_FORMAT,
                image_mime=IMAGE_MIME,
            )

            return Response(
                content=body,
                media_type=f'multipart/mixed; boundary="{boundary}"',
            )

        except RuntimeError as e:
            # Handler raised RuntimeError (e.g., max sessions reached)
            if "Max sessions limit reached" in str(e):
                LOGGER.warning(f"[Service] Connect rejected: {e}")
                raise HTTPException(status_code=503, detail=str(e))
            LOGGER.error(f"[Service] Connect error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            LOGGER.error(f"[Service] Connect error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if acquired and _sem is not None:
                _sem.release()

    @app.post("/call")
    async def call(
        request: Request,
        data: Optional[str] = Form(default=None),
        images: Optional[List[UploadFile]] = File(default=None),
    ):
        """
        Call a method on an existing session.

        Request:
            Content-Type: multipart/form-data
            - data: JSON string with {session_id, method, params}
            - images: optional images

        Response:
            Content-Type: multipart/mixed
            - data: method result
            - images: optional result images
        """
        _auth(request)

        acquired = False
        if _sem is not None:
            try:
                await asyncio.wait_for(_sem.acquire(), timeout=ADMIT_TIMEOUT)
                acquired = True
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail="server busy")

        try:
            data_dict = parse_data_field(data)
            img_list = await read_images(images)

            session_id = data_dict.get("session_id")
            method = data_dict.get("method")
            params = data_dict.get("params", {})
            _log_request_detail("/call", {
                "session_id": session_id,
                "method": method,
                "params_keys": list(params.keys()) if isinstance(params, dict) else str(params),
            }, {"num_images": len(img_list) if img_list else 0})

            if not session_id:
                raise HTTPException(status_code=400, detail="session_id required")
            if not method:
                raise HTTPException(status_code=400, detail="method required")

            result = await handler.call(session_id, method, params, img_list)

            boundary, body = encode_multipart(
                result.data,
                result.images,
                image_format=IMAGE_FORMAT,
                image_mime=IMAGE_MIME,
            )

            return Response(
                content=body,
                media_type=f'multipart/mixed; boundary="{boundary}"',
            )

        except ValueError as e:
            # Handler raised ValueError (e.g., session not found)
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            # Handler raised RuntimeError (e.g., max sessions reached)
            if "Max sessions limit reached" in str(e):
                raise HTTPException(status_code=503, detail=str(e))
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            LOGGER.error(f"[Service] Call error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if acquired and _sem is not None:
                _sem.release()

    return app
