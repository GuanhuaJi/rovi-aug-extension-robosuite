def fast_step(env, action):
    """
    Minimal physics step that **完全绕开渲染与观测**，
    兼容 robosuite 1.0 ~ 1.6 及任何自定义 MujocoEnv。
    返回 (reward, done, info)；不生成 obs。
    """
    # 1. 计算“一个控制周期需要多少 sim 子步”
    substeps = int(env.control_timestep / env.model_timestep)

    # 2. 先把动作写入电机
    policy_step = True
    for _ in range(substeps):
        # 和 robosuite.step() 保持同样的前/后处理
        if hasattr(env, "_pre_action"):
            env._pre_action(action, policy_step=policy_step)
        # 纯物理推进：旧版只有 sim.step()
        if hasattr(env.sim, "step"):
            env.sim.step()
        else:                           # 极早期 robosuite
            env.sim.forward()
        policy_step = False

    # 3. 时间推进
    if hasattr(env, "cur_time"):       # robosuite >=0.4
        env.cur_time += env.control_timestep
    if hasattr(env, "timestep"):       # robosuite <=0.3
        env.timestep += 1
    return 0.0, False, {}