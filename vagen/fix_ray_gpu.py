"""
Fix Ray GPU detection on Blackwell (RTX 50xx) GPUs.

Blackwell GPUs expose /dev/accel0, which Ray's TPU detector picks up
as a TPU device, preventing GPU auto-detection. This module patches
the TPU accelerator manager to return 0 devices so Ray falls through
to the NVIDIA GPU detector instead.

Usage: import this module before any `ray.init()` call, or add
    import vagen.fix_ray_gpu
at the top of your entry point.
"""
import ray._private.accelerators.tpu as _tpu_mod

_tpu_mod.TPUAcceleratorManager.get_current_node_num_accelerators = staticmethod(lambda: 0)
