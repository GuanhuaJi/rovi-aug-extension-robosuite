"""
robot2robot.envs
────────────────
High-level environment wrappers built on top of sim.*

Right now we expose just one public class:
    - SourceEnvWrapper
"""

from .source_env import SourceEnvWrapper
from .target_env import TargetEnvWrapper

__all__ = ["SourceEnvWrapper", "TargetEnvWrapper"]

# (Optional) IDE autocompletion friendly
from importlib import import_module as _imp
globals()["source_env"] = _imp(f"{__name__}.source_env")
del _imp

