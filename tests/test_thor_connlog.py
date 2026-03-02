"""Test AI2-THOR with connection-level logging in werkzeug."""
import time, os, sys, traceback

LOG = "/tmp/thor_connlog_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR connection logging test...")

import werkzeug.serving
log(f"werkzeug version: {werkzeug.__version__}")

# Patch handle_one_request to log connection state
if hasattr(werkzeug.serving.WSGIRequestHandler, 'handle_one_request'):
    original_handle_one_request = werkzeug.serving.WSGIRequestHandler.handle_one_request

    def logged_handle_one_request(self):
        log(f"[WERKZEUG] handle_one_request called, close_connection={self.close_connection}")
        result = original_handle_one_request(self)
        log(f"[WERKZEUG] handle_one_request done, close_connection={self.close_connection}")
        return result

    werkzeug.serving.WSGIRequestHandler.handle_one_request = logged_handle_one_request
else:
    log("[WERKZEUG] No handle_one_request method!")

# Patch handle to log connection lifecycle
original_handle = werkzeug.serving.WSGIRequestHandler.handle

def logged_handle(self):
    log(f"[WERKZEUG] handle() called - new connection from {self.client_address}")
    try:
        result = original_handle(self)
        log(f"[WERKZEUG] handle() completed normally")
        return result
    except Exception as e:
        log(f"[WERKZEUG] handle() exception: {e}")
        raise

werkzeug.serving.WSGIRequestHandler.handle = logged_handle

# Patch run_wsgi to log
original_run_wsgi = werkzeug.serving.WSGIRequestHandler.run_wsgi

def logged_run_wsgi(self):
    log(f"[WERKZEUG] run_wsgi called")
    try:
        result = original_run_wsgi(self)
        log(f"[WERKZEUG] run_wsgi returned: {result}")
        return result
    except Exception as e:
        log(f"[WERKZEUG] run_wsgi exception: {e}")
        raise

werkzeug.serving.WSGIRequestHandler.run_wsgi = logged_run_wsgi

# Also log when wfile is closed/shutdown
import socket
original_socket_shutdown = socket.socket.shutdown

def logged_socket_shutdown(self, how):
    log(f"[SOCKET] socket.shutdown({how}) called on {self.getsockname()} -> {self.getpeername()}")
    import traceback as tb
    log(f"[SOCKET] Traceback: {''.join(tb.format_stack()[-5:-1])}")
    return original_socket_shutdown(self, how)

socket.socket.shutdown = logged_socket_shutdown

original_socket_close = socket.socket.close

def logged_socket_close(self):
    try:
        log(f"[SOCKET] socket.close() called on fd={self.fileno()}")
    except:
        log(f"[SOCKET] socket.close() called (fd unavailable)")
    return original_socket_close(self)

socket.socket.close = logged_socket_close

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

    controller.stop()
    log("Controller stopped!")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
