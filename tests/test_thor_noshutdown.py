"""Test AI2-THOR with patched shutdown_request to prevent premature socket shutdown."""
import time, os, sys, traceback

LOG = "/tmp/thor_noshutdown_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR no-shutdown fix test...")

# Patch: prevent socketserver from calling socket.shutdown(SHUT_WR)
# which causes Unity's Mono to get SocketException
import werkzeug.serving

original_make_server = werkzeug.serving.make_server

def patched_make_server(host, port, app, **kwargs):
    server = original_make_server(host, port, app, **kwargs)
    # Override shutdown_request to skip socket.shutdown(SHUT_WR)
    original_shutdown_request = server.shutdown_request.__func__ if hasattr(server.shutdown_request, '__func__') else None

    def no_shutdown_request(request):
        """Skip socket.shutdown() - go directly to close."""
        server.close_request(request)

    server.shutdown_request = no_shutdown_request
    return server

werkzeug.serving.make_server = patched_make_server
log("Patched werkzeug to skip socket.shutdown()")

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
    log("Controller stopped!")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
