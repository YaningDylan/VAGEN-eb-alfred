"""Directly test EBAlfEnv initialization timing."""
import time, os, sys

LOG = "/tmp/ebalfenv_init_test.log"

def log(msg):
    line = f"[{time.time():.1f}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")

os.environ["DISPLAY"] = ":0"

log("Starting EBAlfEnv test...")

t0 = time.time()
log("Importing EmbodiedBench...")
import embodiedbench.envs.eb_alfred.EBAlfEnv as ebalfenv_mod
ebalfenv_mod.X_DISPLAY = "0"
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
log(f"Import done in {time.time()-t0:.1f}s")

log("Creating EBAlfEnv...")
env = EBAlfEnv(eval_set="base", exp_name="test_init", resolution=500)
log(f"EBAlfEnv created in {time.time()-t0:.1f}s")
log(f"Number of episodes: {env.number_of_episodes}")

log("Calling reset()...")
env.reset()
log(f"Reset done in {time.time()-t0:.1f}s")

log(f"Frame shape: {env.env.last_event.frame.shape}")
log(f"Task: {env.episode_language_instruction}")

env.close()
log(f"All done in {time.time()-t0:.1f}s")
