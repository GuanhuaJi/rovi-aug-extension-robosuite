def _robot_geom_ids(env):
    sim   = env.sim
    robot = env.robots[0]

    # ---- 1. arm geoms ----
    model = getattr(robot, "robot_model", robot)
    arm_names = (getattr(model, "geom_names", []) or
                 getattr(model, "visual_geoms", []) +
                 getattr(model, "contact_geoms", []))

    # ---- 2. gripper geoms ----
    grip = getattr(robot, "gripper", None)
    grip_names = []
    if grip is not None:
        grip_names = getattr(grip, "visual_geoms", []) + getattr(grip, "contact_geoms", [])

    # ---- 3. union + id mapping ----
    names = set(arm_names) | set(grip_names)
    ids = {sim.model.geom_name2id(n) for n in names if n in sim.model.geom_names}
    return ids