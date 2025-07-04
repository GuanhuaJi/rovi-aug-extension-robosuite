"""
robot2robot.sim
───────────────
Thin wrappers around Mujoco / Robosuite objects.

Public surface:
    - CameraWrapper
    - RobotCameraWrapper
    - _robot_geom_ids
"""

# ── 公共 re-exports ──────────────────────────────────────────
from .camera            import CameraWrapper
from .robot_camera      import RobotCameraWrapper
from .geom_utils        import _robot_geom_ids
from .dataset_loader    import gripper_convert, load_states_from_harsha

__all__ = [
    "CameraWrapper",
    "RobotCameraWrapper",
    "_robot_geom_ids",
    "gripper_convert",
    "load_states_from_harsha",
]

# （可选）把子模块本身挂在顶层，便于 IDE 补全
from importlib import import_module as _imp
for _name in ("camera", "robot_camera", "geom_utils", "dataset_loader"):
    globals()[_name] = _imp(f"{__name__}.{_name}")
del _imp, _name
