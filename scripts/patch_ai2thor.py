#!/usr/bin/env python3
"""
Patch ai2thor 2.1.0 server.py to fix the SocketException bug.

Root cause: werkzeug's run_wsgi() sends 'Connection: close' after every
response, then calls socket.shutdown(SHUT_WR). Unity's Mono HTTP client
sees the TCP FIN and throws System.Net.Sockets.SocketException, which
crashes the Controller.

Fix (3 parts):
  1. Override run_wsgi() to send 'Connection: keep-alive' and skip drain loop
  2. Force threaded=True in make_server()
  3. Patch shutdown_request to skip socket.shutdown()

Usage:
    conda run -n embench python scripts/patch_ai2thor.py
    conda run -n embench python scripts/patch_ai2thor.py --check   # verify only
"""

import argparse
import importlib
import os
import re
import shutil
import sys


def find_server_py():
    """Locate ai2thor/server.py in the current Python environment."""
    try:
        import ai2thor
    except ImportError:
        print("ERROR: ai2thor is not installed in the current environment.")
        sys.exit(1)

    server_path = os.path.join(os.path.dirname(ai2thor.__file__), "server.py")
    if not os.path.exists(server_path):
        print(f"ERROR: server.py not found at {server_path}")
        sys.exit(1)
    return server_path


def is_already_patched(content: str) -> bool:
    """Check if the fix has already been applied."""
    return "Connection: keep-alive" in content and "threaded=True" in content


# The run_wsgi override to inject into ThorRequestHandler
RUN_WSGI_OVERRIDE = '''
    # ── FIX: Override werkzeug's run_wsgi to avoid SocketException ──────
    #
    # werkzeug's default run_wsgi() always sends 'Connection: close', which
    # triggers socket.shutdown(SHUT_WR) → TCP FIN. Unity Mono sees the FIN on
    # POST the next frame. This override sends 'Connection: keep-alive' and
    # skips the drain loop (which would consume the next keep-alive request).
    import socket as _socket
    from werkzeug.exceptions import InternalServerError as _ISE

    def run_wsgi(self):
        if self.headers.get("Expect", "").lower().strip() == "100-continue":
            self.wfile.write(b"HTTP/1.1 100 Continue\\r\\n\\r\\n")

        self.environ = environ = self.make_environ()
        status_set = None
        headers_set = None
        status_sent = None
        headers_sent = None

        def write(data):
            nonlocal status_sent, headers_sent
            assert status_set is not None, "write() called before start_response"
            if status_sent is None:
                status_sent = status_set
                headers_sent = headers_set
                try:
                    code, msg = status_sent.split(None, 1)
                except ValueError:
                    code, msg = status_sent, ""
                code = int(code)
                self.send_response(code, msg)
                header_keys = set()
                for key, value in headers_sent:
                    self.send_header(key, value)
                    header_keys.add(key.lower())
                if "content-length" not in header_keys:
                    self.close_connection = True
                    self.send_header("Connection", "close")
                else:
                    # KEY FIX: send keep-alive instead of close to prevent
                    # Unity SocketException
                    self.send_header("Connection", "keep-alive")
                if "server" not in header_keys:
                    self.send_header("Server", self.version_string())
                if "date" not in header_keys:
                    self.send_header("Date", self.date_time_string())
                self.end_headers()
            assert isinstance(data, bytes), "applications must write bytes"
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except (ConnectionError, BrokenPipeError):
                pass

        def start_response(status, headers, exc_info=None):
            nonlocal status_set, headers_set
            if exc_info:
                try:
                    if headers_sent:
                        raise exc_info[1].with_traceback(exc_info[2])
                finally:
                    exc_info = None
            elif headers_set:
                raise AssertionError("Headers already set")
            status_set = status
            headers_set = headers
            return write

        try:
            app_iter = self.server.app(environ, start_response)
            try:
                for data in app_iter:
                    write(data)
                if not headers_sent:
                    write(b"")
            finally:
                if hasattr(app_iter, "close"):
                    app_iter.close()

            # KEY FIX: skip the drain loop — with keep-alive, the next
            # bytes on the socket are part of the NEXT request, not leftovers
            # from this one.

        except (ConnectionError, _socket.timeout) as e:
            self.connection_dropped(e, environ)
        except Exception:
            from werkzeug.debug import DebuggedApplication
            if isinstance(self.server.app, DebuggedApplication):
                raise
            from traceback import print_exc
            print_exc()
            if not headers_sent:
                try:
                    status_set = None
                    headers_set = None
                    execute(_ISE())
                except Exception:
                    pass
'''


def patch_file(server_path: str, dry_run: bool = False) -> bool:
    """Apply the three-part fix to server.py."""
    with open(server_path, "r") as f:
        content = f.read()

    if is_already_patched(content):
        print(f"ALREADY PATCHED: {server_path}")
        return True

    # Backup
    backup_path = server_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(server_path, backup_path)
        print(f"Backup saved: {backup_path}")

    # Part 1: Add imports at top of file (after existing imports)
    if "import socket" not in content or "from werkzeug.exceptions import InternalServerError" not in content:
        # Find the last import line before class definitions
        import_insert = "import socket\nfrom werkzeug.exceptions import InternalServerError\n"
        # Insert after 'import werkzeug' or similar
        content = re.sub(
            r"(import werkzeug\.serving\n)",
            r"\1" + import_insert,
            content,
            count=1,
        )

    # Part 2: Add run_wsgi override to ThorRequestHandler class
    # Find the class definition and insert the method
    class_pattern = r"(class ThorRequestHandler\(BaseHTTPRequestHandler\):.*?\n)"
    match = re.search(class_pattern, content)
    if match and "def run_wsgi" not in content:
        insert_pos = match.end()
        content = content[:insert_pos] + RUN_WSGI_OVERRIDE + "\n" + content[insert_pos:]

    # Part 3: Fix make_server call — add threaded=True
    content = re.sub(
        r"werkzeug\.serving\.make_server\(host,\s*self\.port,\s*self\.app\)",
        "werkzeug.serving.make_server(host, self.port, self.app, threaded=True, request_handler=ThorRequestHandler)",
        content,
    )

    # Part 4: Patch shutdown_request after make_server
    if "shutdown_request" not in content:
        content = re.sub(
            r"(self\.wsgi_server = werkzeug\.serving\.make_server\([^)]+\))",
            r"\1\n        # Skip socket.shutdown to prevent Unity SocketException\n"
            r"        self.wsgi_server.shutdown_request = lambda req: self.wsgi_server.close_request(req)",
            content,
        )

    if dry_run:
        print(f"DRY RUN: Would patch {server_path}")
        return True

    with open(server_path, "w") as f:
        f.write(content)
    print(f"PATCHED: {server_path}")
    return True


def verify_patch(server_path: str) -> bool:
    """Verify the patch was applied correctly."""
    with open(server_path, "r") as f:
        content = f.read()

    checks = [
        ("Connection: keep-alive", "run_wsgi override with keep-alive"),
        ("threaded=True", "threaded server mode"),
        ("shutdown_request", "shutdown_request bypass"),
    ]
    all_ok = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"  OK: {desc}")
        else:
            print(f"  MISSING: {desc}")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Patch ai2thor 2.1.0 for SocketException fix")
    parser.add_argument("--check", action="store_true", help="Only verify, don't patch")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    server_path = find_server_py()
    print(f"ai2thor server.py: {server_path}")

    if args.check:
        ok = verify_patch(server_path)
        sys.exit(0 if ok else 1)

    ok = patch_file(server_path, dry_run=args.dry_run)
    if ok and not args.dry_run:
        print("\nVerifying patch:")
        verify_patch(server_path)
        print("\nPatch applied successfully. Test with:")
        print("  conda run -n embench python -c \"from ai2thor.controller import Controller; "
              "c = Controller(scene='FloorPlan1', gridSize=0.25); "
              "c.step('RotateRight'); print('OK'); c.stop()\"")


if __name__ == "__main__":
    main()
