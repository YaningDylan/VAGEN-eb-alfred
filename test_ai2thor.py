import os, time, sys
os.environ['DISPLAY'] = ':0'
sys.stdout.reconfigure(line_buffering=True)

import ai2thor.controller
t0 = time.time()
print(f'[{time.time()-t0:.1f}s] Starting controller...', flush=True)
c = ai2thor.controller.Controller(
    quality='Very Low',
    player_screen_width=300,
    player_screen_height=300,
    x_display='0',
)
print(f'[{time.time()-t0:.1f}s] Controller started. Reset FloorPlan1...', flush=True)
e = c.reset('FloorPlan1')
print(f'[{time.time()-t0:.1f}s] Reset done, success={e.metadata["lastActionSuccess"]}', flush=True)
c.stop()
print(f'[{time.time()-t0:.1f}s] SUCCESS', flush=True)
