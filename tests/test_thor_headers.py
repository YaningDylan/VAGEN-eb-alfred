"""Test AI2-THOR - log the HTTP headers Unity sends."""
import time, os, sys, traceback

LOG = "/tmp/thor_headers_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR headers test...")

# Patch Flask /train to log request headers
import ai2thor.server as server_mod
from flask import request as flask_request

original_server_init = server_mod.Server.__init__

def patched_server_init(self, request_queue, response_queue, host, port=0, threaded=False):
    original_server_init(self, request_queue, response_queue, host, port, threaded)

    original_train = self.app.view_functions.get('train')
    if original_train:
        counter = {'n': 0}
        def logged_train():
            counter['n'] += 1
            n = counter['n']
            log(f"[FLASK] /train #{n}")
            log(f"[FLASK] HTTP version: {flask_request.environ.get('SERVER_PROTOCOL', '?')}")
            log(f"[FLASK] Connection header: '{flask_request.headers.get('Connection', 'MISSING')}'")
            log(f"[FLASK] All headers:")
            for key, value in flask_request.headers:
                log(f"[FLASK]   {key}: {value}")
            try:
                result = original_train()
                log(f"[FLASK] /train #{n} responded OK")
                return result
            except Exception as e:
                log(f"[FLASK] /train #{n} ERROR: {e}")
                raise
        self.app.view_functions['train'] = logged_train

server_mod.Server.__init__ = patched_server_init

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

    controller.stop()
    log("Controller stopped!")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
