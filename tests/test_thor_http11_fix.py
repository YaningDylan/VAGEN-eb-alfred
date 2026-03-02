"""Test AI2-THOR with proper HTTP/1.1 keep-alive implementation.

Root cause: Unity's Mono sends HTTP/1.1 requests WITHOUT a Connection header.
Python's BaseHTTPRequestHandler only sets close_connection=False when
Connection: keep-alive is explicitly present. Per HTTP/1.1 spec (RFC 7230),
keep-alive is the DEFAULT when no Connection header is sent.

Fix: Override parse_request to implement the spec correctly.
"""
import time, os, sys, traceback

LOG = "/tmp/thor_http11_fix_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR HTTP/1.1 keep-alive fix test...")

# Fix: Patch WSGIRequestHandler to default to keep-alive for HTTP/1.1
import werkzeug.serving

original_parse_request = werkzeug.serving.WSGIRequestHandler.parse_request

def fixed_parse_request(self):
    result = original_parse_request(self)
    if result:
        # HTTP/1.1 default: keep-alive when no Connection header
        conntype = self.headers.get('Connection', '')
        if not conntype and self.request_version >= 'HTTP/1.1':
            self.close_connection = False
    return result

werkzeug.serving.WSGIRequestHandler.parse_request = fixed_parse_request
log("Patched HTTP/1.1 keep-alive default")

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

    controller.stop()
    log("All steps passed! Controller stopped.")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
