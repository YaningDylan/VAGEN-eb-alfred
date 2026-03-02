"""Test AI2-THOR with multiple steps - does the 3rd request always fail?"""
import time, os, sys, traceback

LOG = "/tmp/thor_multistep_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR multi-step test...")

try:
    from ai2thor.controller import Controller

    log("Creating Controller + start()...")
    controller = Controller(quality='MediumCloseFitShadows')
    controller.start(
        player_screen_height=500,
        player_screen_width=500,
        x_display="0"
    )
    log(f"Start done (frame 1). Frame: {controller.last_event.frame.shape}")

    # Try multiple steps WITHOUT reset - just use whatever scene loaded
    log("Step 1: MoveAhead...")
    event = controller.step(dict(action='MoveAhead'))
    log(f"Step 1 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 2: RotateRight...")
    event = controller.step(dict(action='RotateRight'))
    log(f"Step 2 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 3: MoveAhead...")
    event = controller.step(dict(action='MoveAhead'))
    log(f"Step 3 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 4: LookDown...")
    event = controller.step(dict(action='LookDown'))
    log(f"Step 4 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 5: RotateLeft...")
    event = controller.step(dict(action='RotateLeft'))
    log(f"Step 5 done! Success: {event.metadata['lastActionSuccess']}")

    controller.stop()
    log("Controller stopped!")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
