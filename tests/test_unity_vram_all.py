"""Measure GPU memory + CPU/RAM per Unity process at multiple resolutions."""
import os, subprocess, time, sys, psutil
os.environ['DISPLAY'] = ':0'

NUM_INSTANCES = 4

def get_gpu_used_mb():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return int(r.stdout.strip().split('\n')[0])

def get_unity_pids():
    """Find all Unity (thor) processes."""
    pids = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = p.info['name'] or ''
            cmdline = ' '.join(p.info['cmdline'] or [])
            if 'thor-' in name.lower() or 'thor-' in cmdline.lower():
                pids.append(p.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pids

def measure_unity_resources(pids):
    """Measure total CPU% and RSS of Unity processes."""
    total_rss_mb = 0
    for pid in pids:
        try:
            p = psutil.Process(pid)
            mem = p.memory_info()
            total_rss_mb += mem.rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total_rss_mb

from ai2thor.controller import Controller

resolution = int(sys.argv[1]) if len(sys.argv) > 1 else 500

print(f"\n{'='*60}", flush=True)
print(f"Testing resolution: {resolution}x{resolution}", flush=True)
print(f"{'='*60}", flush=True)

baseline_gpu = get_gpu_used_mb()
baseline_pids = get_unity_pids()
print(f"Baseline GPU: {baseline_gpu} MiB", flush=True)

controllers = []
deltas = []
ram_deltas = []

for i in range(NUM_INSTANCES):
    before_gpu = get_gpu_used_mb()
    before_ram = measure_unity_resources(get_unity_pids())
    
    c = Controller()
    c.start()
    c.reset('FloorPlan1')
    if resolution != 300:
        c.step(dict(action='ChangeResolution', x=resolution, y=resolution))
    # Do a few steps to stabilize
    for _ in range(3):
        c.step(dict(action='RotateRight'))
    time.sleep(1)
    
    after_gpu = get_gpu_used_mb()
    after_ram = measure_unity_resources(get_unity_pids())
    
    gpu_delta = after_gpu - before_gpu
    ram_delta = after_ram - before_ram
    deltas.append(gpu_delta)
    ram_deltas.append(ram_delta)
    controllers.append(c)
    print(f"  Instance {i+1}: GPU +{gpu_delta} MiB, RAM +{ram_delta:.0f} MiB", flush=True)

total_gpu = get_gpu_used_mb()
total_ram = measure_unity_resources(get_unity_pids())
num_unity = len(get_unity_pids()) - len(baseline_pids)

avg_gpu = (total_gpu - baseline_gpu) / NUM_INSTANCES
avg_ram = total_ram / max(num_unity, NUM_INSTANCES)

print(f"\n  Summary ({resolution}x{resolution}):", flush=True)
print(f"    Total GPU delta: {total_gpu - baseline_gpu} MiB for {NUM_INSTANCES} instances", flush=True)
print(f"    Avg GPU per instance: {avg_gpu:.0f} MiB", flush=True)
print(f"    Avg RAM per instance: {avg_ram:.0f} MiB", flush=True)
print(f"    Unity processes: {num_unity}", flush=True)

for c in controllers:
    c.stop()
time.sleep(2)

final_gpu = get_gpu_used_mb()
print(f"    After cleanup: {final_gpu} MiB (released {total_gpu - final_gpu} MiB)", flush=True)

# Output machine-readable summary
print(f"\nRESULT|{resolution}|{avg_gpu:.0f}|{avg_ram:.0f}", flush=True)
