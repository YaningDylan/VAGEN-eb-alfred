"""Test AI2-THOR with strace on the Python process itself."""
import time, os, sys, traceback, subprocess, signal

LOG = "/tmp/thor_strace2_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR strace test (Python side)...")

# Start strace on our own process for network syscalls
strace_log = "/tmp/python_strace.log"
strace_proc = subprocess.Popen(
    ["strace", "-f", "-e", "trace=network,write,read,close,shutdown",
     "-p", str(os.getpid()),
     "-o", strace_log],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
time.sleep(0.5)
log(f"strace attached (pid={strace_proc.pid})")

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

# Stop strace
strace_proc.send_signal(signal.SIGINT)
strace_proc.wait()
log("strace stopped")

# Show relevant strace output
log("\n=== STRACE OUTPUT (socket operations) ===")
if os.path.exists(strace_log):
    with open(strace_log) as f:
        lines = f.readlines()
    # Filter for interesting operations
    for line in lines:
        if any(k in line for k in ['shutdown', 'close(', 'connect', 'accept', 'bind', 'listen']):
            log(line.rstrip())

log(f"Test finished in {time.time()-T0:.1f}s")
