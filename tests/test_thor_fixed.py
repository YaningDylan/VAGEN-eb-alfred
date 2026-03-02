"""Test AI2-THOR with the directly patched server.py (no monkey-patching needed)."""
import time, os, traceback

LOG = "/tmp/thor_fixed_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Testing directly patched ai2thor/server.py (no monkey-patching)...")

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

    for i in range(1, 6):
        action = ['MoveAhead', 'RotateRight', 'LookDown', 'RotateLeft', 'LookUp'][i-1]
        log(f"Step {i}: {action}...")
        event = controller.step(dict(action=action))
        log(f"Step {i} done! Success: {event.metadata['lastActionSuccess']}")

    log("Reset FloorPlan1...")
    event = controller.reset('FloorPlan1')
    log(f"Reset done! Frame: {event.frame.shape}")

    log("Initialize...")
    event = controller.step(dict(action='Initialize', gridSize=0.25, renderImage=True, makeAgentsVisible=False))
    log(f"Initialize done! Success: {event.metadata['lastActionSuccess']}")

    for i in range(6, 11):
        action = ['MoveAhead', 'RotateRight', 'LookDown', 'RotateLeft', 'LookUp'][i-6]
        log(f"Step {i}: {action}...")
        event = controller.step(dict(action=action))
        log(f"Step {i} done! Success: {event.metadata['lastActionSuccess']}")

    log("Reset FloorPlan2...")
    event = controller.reset('FloorPlan2')
    log(f"Reset 2 done! Frame: {event.frame.shape}")

    log("Initialize 2...")
    event = controller.step(dict(action='Initialize', gridSize=0.25, renderImage=True, makeAgentsVisible=False))
    log(f"Initialize 2 done! Success: {event.metadata['lastActionSuccess']}")

    for i in range(11, 16):
        action = ['MoveAhead', 'RotateRight', 'LookDown', 'RotateLeft', 'LookUp'][i-11]
        log(f"Step {i}: {action}...")
        event = controller.step(dict(action=action))
        log(f"Step {i} done! Success: {event.metadata['lastActionSuccess']}")

    controller.stop()
    log(f"ALL 15 STEPS + 2 RESETS PASSED! Finished in {time.time()-T0:.1f}s")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
