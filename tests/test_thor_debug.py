"""Test AI2-THOR with debug logging on the Flask server side."""
import time, os, sys, traceback, threading

LOG = "/tmp/thor_debug_test.log"

def log(msg):
    line = f"[{time.time():.1f} | +{time.time()-T0:.1f}s] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

os.environ["DISPLAY"] = ":0"
T0 = time.time()

log("Starting AI2-THOR debug test...")

# Kill any leftover Unity processes
import subprocess
subprocess.run(["pkill", "-9", "-f", "thor-201909"], capture_output=True)
time.sleep(1)

try:
    # Monkey-patch werkzeug to add connection logging
    import werkzeug.serving
    original_handle = werkzeug.serving.WSGIRequestHandler.handle

    def debug_handle(self):
        log(f"[WERKZEUG] New connection from {self.client_address}")
        try:
            result = original_handle(self)
            log(f"[WERKZEUG] Connection handled OK from {self.client_address}")
            return result
        except Exception as e:
            log(f"[WERKZEUG] Connection error from {self.client_address}: {e}")
            raise

    werkzeug.serving.WSGIRequestHandler.handle = debug_handle

    # Also patch the run_wsgi to log
    original_run_wsgi = werkzeug.serving.WSGIRequestHandler.run_wsgi

    def debug_run_wsgi(self):
        log(f"[WERKZEUG] run_wsgi called, method={getattr(self, 'command', '?')}, path={getattr(self, 'path', '?')}")
        try:
            result = original_run_wsgi(self)
            log(f"[WERKZEUG] run_wsgi completed OK")
            return result
        except Exception as e:
            log(f"[WERKZEUG] run_wsgi error: {e}")
            raise

    werkzeug.serving.WSGIRequestHandler.run_wsgi = debug_run_wsgi

    # Patch make_server to log the port
    original_make_server = werkzeug.serving.make_server

    def debug_make_server(host, port, app, **kwargs):
        log(f"[WERKZEUG] make_server({host}, {port}, threaded={kwargs.get('threaded')})")
        server = original_make_server(host, port, app, **kwargs)
        actual_port = server.server_address[1]
        log(f"[WERKZEUG] Server listening on {host}:{actual_port}")
        return server

    werkzeug.serving.make_server = debug_make_server

    # Now import and use ai2thor
    from ai2thor.controller import Controller

    # Monitor network connections in background
    def monitor_connections():
        while True:
            try:
                result = subprocess.run(
                    ["ss", "-tnp"],
                    capture_output=True, text=True, timeout=5
                )
                conns = [l for l in result.stdout.split('\n')
                         if 'thor' in l or 'python' in l]
                if conns:
                    log(f"[MONITOR] Active connections: {len(conns)}")
                    for c in conns[:5]:
                        log(f"[MONITOR]   {c.strip()}")
            except:
                pass
            time.sleep(3)

    monitor = threading.Thread(target=monitor_connections, daemon=True)
    monitor.start()

    log("Creating Controller...")
    controller = Controller(quality='MediumCloseFitShadows')
    log("Calling controller.start()...")
    controller.start(
        player_screen_height=500,
        player_screen_width=500,
        x_display="0"
    )
    log("Controller started successfully!")
    log(f"Frame shape: {controller.last_event.frame.shape}")

    controller.stop()
    log("Controller stopped!")

except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())

log(f"Test finished in {time.time()-T0:.1f}s")
