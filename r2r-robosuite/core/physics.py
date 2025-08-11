def fast_step(env, action):
    """
    Minimal physics step that **completely bypasses rendering and observation**, 
    compatible with robosuite 1.0 ~ 1.6 and any custom MujocoEnv.
    Returns (reward, done, info); no obs produced.
    """
    # 1. Calculate how many sim substeps a control cycle requires
    substeps = int(env.control_timestep / env.model_timestep)

    # 2. Write the action to the motors first
    policy_step = True
    for _ in range(substeps):
        # Keep the same pre/post processing as robosuite.step()
        if hasattr(env, "_pre_action"):
            env._pre_action(action, policy_step=policy_step)
        # Pure physics stepping: older versions only have sim.step()
        if hasattr(env.sim, "step"):
            env.sim.step()
        else:                           # very early robosuite
            env.sim.forward()
        policy_step = False

    # 3. Advance time
    if hasattr(env, "cur_time"):       # robosuite >=0.4
        env.cur_time += env.control_timestep
    if hasattr(env, "timestep"):       # robosuite <=0.3
        env.timestep += 1
    return 0.0, False, {}

