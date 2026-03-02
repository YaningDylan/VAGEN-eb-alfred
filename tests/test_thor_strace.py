"""Test AI2-THOR with socket-level tracing."""
import time, os, sys, traceback, subprocess

LOG = "/tmp/thor_strace_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR strace test...")

try:
    # Import ai2thor directly
    from ai2thor.controller import Controller
    import ai2thor.controller as ctrl_mod

    # Monkey-patch to add strace to Unity process
    original_popen = subprocess.Popen
    unity_pid = None

    class StracePopen:
        """Wrapper to strace the Unity process."""
        def __init__(self, cmd, **kwargs):
            log(f"Popen called with: {cmd[:3]}...")
            # Check if this is the Unity process
            if any("thor-201909" in str(c) for c in cmd):
                log("Detected Unity process launch - adding network trace")
                # Run strace on network syscalls
                strace_cmd = ["strace", "-f", "-e", "trace=network", "-o", "/tmp/unity_strace.log"] + list(cmd)
                self._proc = original_popen(strace_cmd, **kwargs)
                log(f"Unity+strace PID: {self._proc.pid}")
            else:
                self._proc = original_popen(cmd, **kwargs)

        def __getattr__(self, name):
            return getattr(self._proc, name)

    subprocess.Popen = StracePopen

    log("Creating Controller...")
    controller = Controller(
        quality='MediumCloseFitShadows',
        player_screen_height=500,
        player_screen_width=500,
        x_display="0"
    )
    log("Controller created!")

    log(f"Frame shape: {controller.last_event.frame.shape}")

    controller.stop()
    log("Controller stopped!")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")

# Show strace output
if os.path.exists("/tmp/unity_strace.log"):
    log("\n=== STRACE OUTPUT (last 50 lines) ===")
    with open("/tmp/unity_strace.log") as f:
        lines = f.readlines()
        for line in lines[-50:]:
            log(line.rstrip())
