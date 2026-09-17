"""The CF planner service's HTTP door. Standard library only.

    POST /estimate   cf-planner.md §3a/§3b
    GET  /version    §3c

**It stores nothing from a request and calls nothing outside itself** (§6). `route` is the whole of
the behaviour; the server class below only moves bytes to and from it.

Run:  python3 -m service.app          (from the repository root; PORT defaults to 8080)
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from service import planner_service as ps

#: A request is a job and a handful of tiers; anything past this is not one.
MAX_BODY_BYTES = 1 << 20


def route(method, path, body, commit):
    """`(status, payload)` for one request. Pure: no I/O beyond the worker's own table reads."""
    if path == "/version":
        if method != "GET":
            return 405, {"error": "GET only"}
        return 200, ps.version(commit)
    if path == "/estimate":
        if method != "POST":
            return 405, {"error": "POST only"}
        try:
            request = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return 400, {"refused": {"field": "request", "message": "body is not JSON"}}
        try:
            entries = ps.estimate_core(request, commit=commit)
        except ps.Refusal as refusal:
            return 400, {"refused": {"field": refusal.field, "message": refusal.message}}
        except ps.TableUnusable as broken:
            # The service's own committed table, not the caller's request: 503, never a 400.
            return 503, {"error": "service tables unusable", "detail": str(broken)}
        return 200, {"tiers": [ps.project(e) for e in entries]}
    return 404, {"error": "not found"}


def make_handler(commit):
    class Handler(BaseHTTPRequestHandler):
        server_version = "cf-planner"

        def _respond(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, method):
            length = self.headers.get("Content-Length")
            try:
                size = int(length) if length else 0
            except ValueError:
                return self._respond(400, {"error": "bad Content-Length"})
            if size < 0 or size > MAX_BODY_BYTES:
                return self._respond(413, {"error": "body too large"})
            body = self.rfile.read(size) if size else b""
            try:
                status, payload = route(method, self.path.split("?", 1)[0], body, commit)
            except Exception as error:  # noqa: BLE001 — a crash is reported, never a hung socket
                status, payload = 500, {"error": "internal", "type": type(error).__name__}
                print("estimate failed: {!r}".format(error), file=sys.stderr)
            self._respond(status, payload)

        def do_GET(self):  # noqa: N802
            self._handle("GET")

        def do_POST(self):  # noqa: N802
            self._handle("POST")

    return Handler


def main():
    commit = ps.service_commit(os.environ)
    # **Both committed tables load before the port opens**, so a broken deploy fails to start
    # instead of answering 500 to every request.
    ps.version(commit)
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(commit))
    print("cf-planner on :{} commit={}".format(port, commit), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
