"""Direct EBAlfEnv test v2 - detailed logging to file."""
import time, os, sys, traceback

LOG = "/tmp/ebalfenv_v2.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting EBAlfEnv test v2...")

try:
    log("Importing EmbodiedBench...")
    import embodiedbench.envs.eb_alfred.EBAlfEnv as ebalfenv_mod
    ebalfenv_mod.X_DISPLAY = "0"
    from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
    log("Import done.")

    log("Creating EBAlfEnv(eval_set='base', resolution=500)...")
    env = EBAlfEnv(eval_set="base", exp_name="test_v2", resolution=500)
    log(f"EBAlfEnv created. Episodes: {env.number_of_episodes}")

    log("Setting episode to 0...")
    env._current_episode_num = 0

    log("Calling env.reset()...")
    env.reset()
    log("Reset done!")

    log(f"Frame: {env.env.last_event.frame.shape}")
    log(f"Task: {env.episode_language_instruction}")

    log("Calling env.close()...")
    env.close()
    log("Close done!")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
