"""Test EBAlfEnv with the fixed ai2thor server.py."""
import time, os, sys, traceback

LOG = "/tmp/ebalfenv_fixed_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Testing EBAlfEnv with fixed server.py...")

# Add EmbodiedBench to path
sys.path.insert(0, '/home/march/workspace/Yaning/Embodied-Reasoning-Agent/eval/EmbodiedBench')

# Override X_DISPLAY before importing EBAlfEnv
import embodiedbench.envs.eb_alfred.EBAlfEnv as ebalfenv_mod
ebalfenv_mod.X_DISPLAY = '0'

try:
    from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv

    log("Creating EBAlfEnv (resolution=500)...")
    env = EBAlfEnv(resolution=500)
    log(f"EBAlfEnv created")

    log("Resetting env (FloorPlan28, trial T20190906_224832_838948)...")
    obs, info = env.reset(
        scene_name='FloorPlan28',
        trial_name='trial_T20190906_224832_838948'
    )
    log(f"Reset done! obs keys: {list(obs.keys())}")
    log(f"Image shape: {obs['image'].shape if hasattr(obs['image'], 'shape') else type(obs['image'])}")

    # Try a few steps
    for i, action in enumerate(["MoveAhead", "RotateRight", "LookDown", "MoveAhead"]):
        log(f"Step {i+1}: {action}...")
        obs, reward, done, truncated, info = env.step(action)
        log(f"Step {i+1} done! reward={reward}, done={done}")

    log("Closing env...")
    env.close()
    log(f"EBAlfEnv test PASSED in {time.time()-T0:.1f}s")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
