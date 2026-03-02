"""Test AI2-THOR with forced keep-alive to fix SocketException."""
import time, os, sys, traceback

LOG = "/tmp/thor_keepalive_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR keep-alive fix test...")

# Patch werkzeug to force keep-alive connections
import werkzeug.serving

original_handle = werkzeug.serving.WSGIRequestHandler.handle

def keepalive_handle(self):
    """Force keep-alive by resetting close_connection after each request."""
    self.close_connection = True
    self.handle_one_request()
    while not self.close_connection:
        self.handle_one_request()

# Override handle_one_request to force keep-alive for HTTP/1.1
original_handle_one = werkzeug.serving.WSGIRequestHandler.handle_one_request

def keepalive_handle_one_request(self):
    result = original_handle_one(self)
    # Force keep-alive for HTTP/1.1 protocol
    if self.protocol_version >= 'HTTP/1.1':
        self.close_connection = False
    return result

werkzeug.serving.WSGIRequestHandler.handle_one_request = keepalive_handle_one_request
log("Patched werkzeug for keep-alive")

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

    controller.stop()
    log("Controller stopped!")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
