"""Test AI2-THOR with detailed Flask handler logging."""
import time, os, sys, traceback, threading

LOG = "/tmp/thor_trace_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR trace test...")

import werkzeug
log(f"werkzeug version: {werkzeug.__version__}")

# Patch the ai2thor Server to add logging
import ai2thor.server as server_mod

original_server_init = server_mod.Server.__init__

def patched_server_init(self, request_queue, response_queue, host, port=0, threaded=False):
    original_server_init(self, request_queue, response_queue, host, port, threaded)

    # Wrap the /train route to add logging
    original_train = None
    for rule in self.app.url_map.iter_rules():
        if rule.rule == '/train':
            original_train = self.app.view_functions[rule.endpoint]
            break

    if original_train:
        from flask import request as flask_request
        train_counter = {'count': 0}

        def logged_train():
            train_counter['count'] += 1
            n = train_counter['count']
            log(f"[FLASK] /train #{n} received, content_length={flask_request.content_length}")
            try:
                result = original_train()
                log(f"[FLASK] /train #{n} responded, status={result.status}")
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

    log("Step 1: MoveAhead...")
    event = controller.step(dict(action='MoveAhead'))
    log(f"Step 1 done! Success: {event.metadata['lastActionSuccess']}")

    log("Step 2: RotateRight...")
    event = controller.step(dict(action='RotateRight'))
    log(f"Step 2 done!")

    controller.stop()
    log("Controller stopped!")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
