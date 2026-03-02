"""Comprehensive fix for AI2-THOR SocketException.

Root cause chain:
1. werkzeug's run_wsgi() ALWAYS sends "Connection: close" header (line 293)
2. BaseHTTPRequestHandler.send_header() sets close_connection=True
3. After handle() returns, socketserver calls socket.shutdown(SHUT_WR)
4. Unity's old Mono (2018) gets SocketException when trying to send next POST

Fix approach:
1. Override run_wsgi in ThorRequestHandler to send "Connection: keep-alive"
   instead of "Connection: close", and skip the drain loop
2. Make server threaded so new connections during Reset can be accepted
3. Properly handle dead connections (don't force keep-alive on closed sockets)
"""
import time, os, sys, traceback, io, selectors, socket

LOG = "/tmp/thor_comprehensive_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting comprehensive AI2-THOR fix test...")

# ============================================================
# Fix 1: Patch ThorRequestHandler.run_wsgi to support keep-alive
# ============================================================
import werkzeug.serving
from werkzeug._internal import _log, _wsgi_encoding_dance
from werkzeug.exceptions import InternalServerError

log(f"werkzeug version: {werkzeug.__version__}")

import ai2thor.server as server_mod

# Save original run_wsgi
_original_wsgi_run_wsgi = werkzeug.serving.WSGIRequestHandler.run_wsgi

def keepalive_run_wsgi(self):
    """Modified run_wsgi that supports HTTP/1.1 keep-alive.

    Changes from werkzeug's original:
    1. Sends "Connection: keep-alive" instead of "Connection: close"
    2. Skips the drain loop that would consume the next request's data
    3. Sets close_connection = False so handle() loops for more requests
    """
    if self.headers.get("Expect", "").lower().strip() == "100-continue":
        self.wfile.write(b"HTTP/1.1 100 Continue\r\n\r\n")

    self.environ = environ = self.make_environ()
    status_set = None
    headers_set = None
    status_sent = None
    headers_sent = None
    chunk_response = False

    def write(data):
        nonlocal status_sent, headers_sent, chunk_response
        assert status_set is not None, "write() before start_response"
        assert headers_set is not None, "write() before start_response"
        if status_sent is None:
            status_sent = status_set
            headers_sent = headers_set
            try:
                code_str, msg = status_sent.split(None, 1)
            except ValueError:
                code_str, msg = status_sent, ""
            code = int(code_str)
            self.send_response(code, msg)
            header_keys = set()
            for key, value in headers_sent:
                self.send_header(key, value)
                header_keys.add(key.lower())

            if (
                not (
                    "content-length" in header_keys
                    or environ["REQUEST_METHOD"] == "HEAD"
                    or (100 <= code < 200)
                    or code in {204, 304}
                )
                and self.protocol_version >= "HTTP/1.1"
            ):
                chunk_response = True
                self.send_header("Transfer-Encoding", "chunked")

            # FIX: Send keep-alive instead of close for HTTP/1.1
            if self.request_version >= 'HTTP/1.1':
                self.send_header("Connection", "keep-alive")
            else:
                self.send_header("Connection", "close")
            self.end_headers()

        assert isinstance(data, bytes), "applications must write bytes"

        if data:
            if chunk_response:
                self.wfile.write(hex(len(data))[2:].encode())
                self.wfile.write(b"\r\n")
            self.wfile.write(data)
            if chunk_response:
                self.wfile.write(b"\r\n")

        self.wfile.flush()

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

    def execute(app):
        application_iter = app(environ, start_response)
        try:
            for data in application_iter:
                write(data)
            if not headers_sent:
                write(b"")
            if chunk_response:
                self.wfile.write(b"0\r\n\r\n")
        finally:
            # FIX: Skip the drain loop for keep-alive connections.
            # The drain loop reads ALL remaining data, which would
            # consume the next request in a keep-alive connection.
            if hasattr(application_iter, "close"):
                application_iter.close()

    try:
        execute(self.server.app)
    except (ConnectionError, socket.timeout) as e:
        self.connection_dropped(e, environ)
    except Exception as e:
        if self.server.passthrough_errors:
            raise

        if status_sent is not None and chunk_response:
            self.close_connection = True

        try:
            if status_sent is None:
                status_set = None
                headers_set = None
            execute(InternalServerError())
        except Exception:
            pass

        from werkzeug.debug.tbtools import DebugTraceback
        msg = DebugTraceback(e).render_traceback_text()
        self.server.log("error", f"Error on request:\n{msg}")


# Apply the patch to ThorRequestHandler
server_mod.ThorRequestHandler.run_wsgi = keepalive_run_wsgi
log("Patched ThorRequestHandler.run_wsgi for keep-alive")

# ============================================================
# Fix 2: Make server threaded
# ============================================================
original_server_init = server_mod.Server.__init__

def threaded_server_init(self, request_queue, response_queue, host, port=0, threaded=False):
    original_server_init(self, request_queue, response_queue, host, port, threaded=True)

server_mod.Server.__init__ = threaded_server_init
log("Patched Server to be threaded")

# ============================================================
# Fix 3: Patch shutdown_request to not call socket.shutdown()
# ============================================================
import socketserver

original_shutdown_request = socketserver.TCPServer.shutdown_request

def safe_shutdown_request(self, request):
    """Skip socket.shutdown(SHUT_WR) - just close the socket.

    socket.shutdown(SHUT_WR) sends a TCP FIN that confuses Unity's Mono.
    """
    self.close_request(request)

socketserver.TCPServer.shutdown_request = safe_shutdown_request
log("Patched shutdown_request to skip socket.shutdown()")

# ============================================================
# Test
# ============================================================
try:
    from ai2thor.controller import Controller

    log("Creating Controller + start()...")
    controller = Controller(quality='MediumCloseFitShadows')
    controller.start(
        player_screen_height=500,
        player_screen_width=500,
        x_display="0"
    )
    log(f"Start done. Frame: {controller.last_event.frame.shape}")

    log("Step 1: MoveAhead...")
    event = controller.step(dict(action='MoveAhead'))
    log(f"Step 1 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 2: RotateRight...")
    event = controller.step(dict(action='RotateRight'))
    log(f"Step 2 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 3: LookDown...")
    event = controller.step(dict(action='LookDown'))
    log(f"Step 3 done! Success: {event.metadata['lastActionSuccess']}")

    log("Reset FloorPlan1...")
    event = controller.reset('FloorPlan1')
    log(f"Reset done! Frame: {event.frame.shape}")

    log("Step 4: Initialize...")
    event = controller.step(dict(
        action='Initialize',
        gridSize=0.25,
        renderImage=True,
        makeAgentsVisible=False,
    ))
    log(f"Initialize done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 5: MoveAhead...")
    event = controller.step(dict(action='MoveAhead'))
    log(f"Step 5 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 6: RotateLeft...")
    event = controller.step(dict(action='RotateLeft'))
    log(f"Step 6 done! Success: {event.metadata['lastActionSuccess']}")

    log("Reset FloorPlan2...")
    event = controller.reset('FloorPlan2')
    log(f"Reset 2 done! Frame: {event.frame.shape}")

    log("Step 7: Initialize...")
    event = controller.step(dict(
        action='Initialize',
        gridSize=0.25,
        renderImage=True,
        makeAgentsVisible=False,
    ))
    log(f"Initialize 2 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 8: MoveAhead...")
    event = controller.step(dict(action='MoveAhead'))
    log(f"Step 8 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 9: RotateRight...")
    event = controller.step(dict(action='RotateRight'))
    log(f"Step 9 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 10: LookUp...")
    event = controller.step(dict(action='LookUp'))
    log(f"Step 10 done! Success: {event.metadata['lastActionSuccess']}")

    controller.stop()
    log("ALL 10 STEPS + 2 RESETS PASSED! Controller stopped.")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
