"""Quick test: how long does AI2-THOR take to initialize?"""
import time, os, sys

LOG = "/tmp/thor_init_test.log"

def log(msg):
    line = f"[{time.time():.1f}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
log("Starting AI2-THOR test...")

t0 = time.time()
log("Importing ai2thor...")
import ai2thor.controller
log(f"Import done in {time.time()-t0:.1f}s")

log("Creating Controller...")
c = ai2thor.controller.Controller()
log(f"Controller created in {time.time()-t0:.1f}s")

log("Resetting to FloorPlan1...")
c.reset("FloorPlan1")
log(f"Reset done in {time.time()-t0:.1f}s")
log(f"Frame shape: {c.last_event.frame.shape}")

c.stop()
log(f"All done in {time.time()-t0:.1f}s")
