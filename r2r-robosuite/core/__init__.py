"""
robot2robot.core
────────────────
Cross-simulator, pure-Python utilities.

This namespace re-exports the handful of helpers you’re expected to call
directly.  Internal helpers stay hidden unless you import sub-modules
explicitly.

Example
-------
>>> from robot2robot.core import fast_step, pick_best_gpu
"""

# ──────────────────────────────────────────────────────────
# Public re-exports
# ──────────────────────────────────────────────────────────
from .gpu       import pick_best_gpu
from .physics   import fast_step
from .geometry  import quat_dist_rad, compute_pose_error
from .signal import smooth_xyz_spikes, reach_further
from .io       import locked_json, atomic_write_json

__all__ = [
    "pick_best_gpu",
    "fast_step",
    "quat_dist_rad",
    "compute_pose_error",
    "load_blacklist",
    "ensure_dir",
    "smooth_xyz_spikes",
    "reach_further",
]

# （可选）让 IDE / REPL 补全时能看到子模块本身
from importlib import import_module as _imp
for _name in ("gpu", "physics", "geometry", "io"):
    globals()[_name] = _imp(f"{__name__}.{_name}")
del _imp, _name
