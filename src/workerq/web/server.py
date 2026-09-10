"""A loopback HTTP server for the local UI.

Standard library only. The tool ships with Typer and Rich and nothing else,
and that restraint is worth keeping for something whose whole job is to stop
this machine falling over: a dashboard is not a reason to add a web framework
and its dependency tree to the thing that guards the queue.

**Loopback only.** The dispatcher has no socket precisely so that worker-q
never runs an unauthenticated network server. A read-mostly UI does not breach
that as long as it cannot be reached from off the machine, so binding anything
but a loopback address is refused rather than warned about.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from workerq import __version__
from workerq.config import Config
from workerq.web.api import Api

#: 8787 is a common choice and was already taken on the machine this was
#: built for, which is exactly the kind of collision a default should avoid.
DEFAULT_PORT = 7676

STATIC_DIR = Path(__file__).parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
}

#: Filters that may be repeated in a query string, and so arrive as lists.
_MULTI = frozenset({"state", "project", "node", "priority", "kind"})


class LoopbackOnly(ValueError):
    """Raised when asked to bind an address the outside world could reach."""


def _require_loopback(hostname: str) -> str:
    try:
        if not ip_address(hostname).is_loopback:
            raise LoopbackOnly(
                f"{hostname} is reachable from outside this machine. The web UI "
                "has no authentication and can cancel jobs, so it only binds "
                "loopback addresses (127.0.0.1 or ::1)."
            )
    except ValueError as exc:
        if isinstance(exc, LoopbackOnly):
            raise
        if hostname != "localhost":
            raise LoopbackOnly(
                f"{hostname!r} is not a loopback address. The web UI has no "
                "authentication, so it only binds 127.0.0.1, ::1 or localhost."
            ) from exc
    return hostname


class Handler(BaseHTTPRequestHandler):
    server_version = f"worker-q/{__version__}"
    api: Api

    # -- plumbing ----------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # BaseHTTPRequestHandler writes to stderr by default, which under
        # pythonw is None and in a terminal is a wall of noise.
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page only ever talks to itself, and saying so keeps a stray
        # third-party script on another local port from reading the queue.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass  # the tab was closed mid-response

    def _json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json({"error": message}, code=code)

    def _query(self) -> dict[str, Any]:
        raw = parse_qs(urlparse(self.path).query, keep_blank_values=False)
        out: dict[str, Any] = {}
        for key, values in raw.items():
            out[key] = values if key in _MULTI else values[-1]
        return out

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) or {}
        except (ValueError, UnicodeDecodeError):
            return {}

    # -- routing -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path.startswith("/api/"):
            try:
                self._json(self._route_api(path))
            except FileNotFoundError as exc:
                self._error(404, str(exc))
            except (KeyError, ValueError) as exc:
                self._error(400, str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                self._error(500, f"{type(exc).__name__}: {exc}")
            return
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        parts = [p for p in path.split("/") if p]
        # /api/jobs/<id>/<action>
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs":
            try:
                result = self.api.act(int(parts[2]), parts[3], self._body())
            except ValueError as exc:
                self._error(400, str(exc))
            except Exception as exc:
                self._error(500, f"{type(exc).__name__}: {exc}")
            else:
                self._json(result)
            return
        self._error(404, "no such endpoint")

    def _route_api(self, path: str) -> Any:
        parts = [p for p in path.split("/") if p][1:]  # drop "api"
        query = self._query()
        if not parts:
            raise FileNotFoundError("no such endpoint")
        head = parts[0]

        if head == "overview":
            return self.api.overview()
        if head == "facets":
            return self.api.facets()
        if head == "accuracy":
            return self.api.accuracy(query)
        if head == "machines":
            return self.api.machines(query)
        if head == "efficiency":
            return self.api.efficiency(query)
        if head == "events":
            return self.api.events(query)
        if head == "jobs":
            if len(parts) == 1:
                return self.api.jobs(query)
            job_id = int(parts[1])
            if len(parts) == 2:
                return self.api.job(job_id)
            if parts[2] == "series":
                return self.api.job_series(job_id)
            if parts[2] == "log":
                return self.api.job_log(job_id, int(query.get("offset") or 0))
        raise FileNotFoundError(f"no such endpoint: {path}")

    def _serve_static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        try:
            # Never serve outside the bundled directory, whatever the URL says.
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._error(403, "outside the static root")
            return
        if not target.is_file():
            # Any unknown path is the app shell: the UI routes on the hash, so
            # a deep link should still load rather than 404.
            target = STATIC_DIR / "index.html"
            if not target.is_file():
                self._error(404, "UI assets are missing from this install")
                return
        content_type = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), content_type)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], api: Api) -> None:
        handler = type("BoundHandler", (Handler,), {"api": api})
        super().__init__(address, handler)
        self.api = api

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            # Each request runs on its own thread and every SQLite handle
            # belongs to the thread that opened it, so the handles have to go
            # when the thread does - otherwise a busy page leaks a connection
            # per request and pins the WAL.
            self.api.close_thread()


def serve(
    config: Config,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: bool = False,
    ready: threading.Event | None = None,
) -> Server:
    """Start the UI. Returns the server; the caller decides how long it runs."""
    _require_loopback(host)
    api = Api(config)
    server = Server((host, port), api)
    if open_browser:
        actual = server.server_address[1]
        threading.Timer(
            0.4, lambda: webbrowser.open(f"http://{host}:{actual}/")
        ).start()
    if ready is not None:
        ready.set()
    return server
