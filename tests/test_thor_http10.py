"""Test AI2-THOR with HTTP/1.0 to avoid keep-alive issues."""
import time, os, sys, traceback

LOG = "/tmp/thor_http10_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR HTTP/1.0 test...")

try:
    # Patch werkzeug to use HTTP/1.0 BEFORE importing ai2thor
    import werkzeug.serving
    werkzeug.serving.WSGIRequestHandler.protocol_version = 'HTTP/1.0'
    log("Patched werkzeug to HTTP/1.0")

    from ai2thor.controller import Controller

    log("Creating Controller + start()...")
    controller = Controller(quality='MediumCloseFitShadows')
    controller.start(
        player_screen_height=500,
        player_screen_width=500,
        x_display="0"
    )
    log(f"Start done. Frame: {controller.last_event.frame.shape}")

    log("Calling controller.reset('FloorPlan1')...")
    event = controller.reset('FloorPlan1')
    log(f"Reset done! Frame: {event.frame.shape}")

    log("Calling controller.step(Initialize)...")
    event = controller.step(dict(
        action='Initialize',
        gridSize=0.25,
        renderImage=True,
        renderDepthImage=False,
        renderClassImage=False,
        renderObjectImage=False,
        makeAgentsVisible=False,
    ))
    log(f"Initialize done! Success: {event.metadata['lastActionSuccess']}")
    log(f"Frame: {event.frame.shape}")

    log("Calling controller.step(MoveAhead)...")
    event = controller.step(dict(action='MoveAhead'))
    log(f"MoveAhead done! Success: {event.metadata['lastActionSuccess']}")

    log("Calling controller.step(RotateRight)...")
    event = controller.step(dict(action='RotateRight'))
    log(f"RotateRight done! Success: {event.metadata['lastActionSuccess']}")

    controller.stop()
    log("Controller stopped!")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
