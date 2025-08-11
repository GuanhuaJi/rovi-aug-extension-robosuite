def gripper_convert(gripper_state_value, robot_type):
    if robot_type == "autolab_ur5":
        return gripper_state_value == 0
    elif robot_type == "furniture_bench":
        return gripper_state_value > 0.05
    elif robot_type == "viola":
        return gripper_state_value > 0.07 # changed
    elif robot_type == "austin_sailor":
        return gripper_state_value > 0.07 # changed
    elif robot_type == "austin_mutex":
        return gripper_state_value > 0.07 # changed
    elif robot_type == "austin_buds":
        return gripper_state_value > 0.07 # changed
    elif robot_type == "nyu_franka":
        return gripper_state_value >= 0
    elif robot_type == "ucsd_kitchen_rlds":
        return gripper_state_value > 0.5
    elif robot_type == "taco_play":
        return gripper_state_value < 0
    elif robot_type == "iamlab_cmu":
        return gripper_state_value > 0.5
    elif robot_type == "toto":
        return gripper_state_value > 0
    elif robot_type == "can":
        return gripper_state_value
    elif robot_type == "lift":
        return gripper_state_value
    elif robot_type == "square":
        return gripper_state_value
    elif robot_type == "stack":
        return gripper_state_value
    elif robot_type == "three_piece_assembly":
        return gripper_state_value
    elif robot_type == "asu_table_top_rlds":
        return gripper_state_value < 0
    elif robot_type == "utokyo_pick_and_place":
        return gripper_state_value > 0.02
    print("UNKNOWN GRIPPER")
    return None

def load_states_from_harsha(robot_dataset, episode, robot_name):
    info_path = Path(harsha_dataset_path[robot_dataset]) / str(episode) / f"panda_replay_info_{episode}.npz"
    info = np.load(info_path, allow_pickle=True)
    joint_angles = info["joint_positions"]
    gripper_states = info["gripper_dist"]
    print(gripper_states)
    translation = info["translation"]
    return joint_angles, gripper_states, translation