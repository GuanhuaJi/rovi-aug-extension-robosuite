import tensorflow_datasets as tfds
import tensorflow as tf


def process_step_toto(step):
    joint_states = step['observation']['state']
    gripper_dist = tf.cond(
        step['action']['open_gripper'],
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states, gripper_dist], axis=0), image
    
def process_step_nyu_franka(step):
    joint_states = step['observation']['state']
    gripper_dist = step['action'][13]
    gripper_dist = tf.cond(
        gripper_dist > 0,
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32),
    )
    
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states[:7], gripper_dist], axis = 0), image

def process_step_autolab_ur5(step):
    joint_states = step['observation']['robot_state'][:14] # First 6 are joints
    gripper_dist = step['observation']['robot_state'][-2]
    gripper_dist = tf.cast(gripper_dist > 0.5, tf.bool)  # Assuming threshold 0.5 for True/False
    gripper_dist = tf.cond(gripper_dist, 
                                lambda: tf.constant([1], dtype=tf.float32), 
                                lambda: tf.constant([0], dtype=tf.float32))
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states[:6], gripper_dist], axis = 0), image

def process_step_ucsd_kitchen_rlds(step):
    joint_states = step['observation']['state'][:7] # First seven joints
    gripper_dist = step["action"][6]
    gripper_dist = tf.cast(gripper_dist < 0.5, tf.bool)
    gripper_dist = tf.cond(
        gripper_dist,
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32)
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states, gripper_dist], axis = 0), image

def process_step_utokyo_pick_and_place(step):
    joint_states = step['observation']['joint_state'][:7] # First seven joints
    gripper_dist = tf.cast(step["action"][-1] > 0.5, tf.bool)
    gripper_dist = tf.cond(
        gripper_dist,
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32)
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states, gripper_dist], axis = 0), image

def process_step_asu_table_top_rlds(step):
    joint_states = step['observation']['state'][:6] # First six joints
    gripper_dist = tf.cast(step['observation']['state'][-1] > 0.2, tf.bool)
    gripper_dist = tf.cond(gripper_dist, 
                                lambda: tf.constant([1], dtype=tf.float32), 
                                lambda: tf.constant([0], dtype=tf.float32))

    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states, gripper_dist], axis = 0), image

def process_step_kaist(step):
    joint_states = step['observation']['state'][:14:2] # First six joints
    gripper_dist = tf.cast(False, tf.bool) # Gripper does not open
    gripper_dist = tf.cond(
        gripper_dist,
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states, gripper_dist], axis = 0), image

def process_step_austin_buds(step):
    joint_states = step['observation']['state'][:7] # franka 
    gripper_dist = step["observation"]["state"][7]
    gripper_dist = tf.cast(gripper_dist > 0.04, tf.bool)
    gripper_dist = tf.cond(
        gripper_dist,
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states[:7], gripper_dist], axis = 0), image

def process_step_austin_sailor(step):
    joint_states = step['observation']['state_joint'] # franka 
    gripper_dist = step["observation"]["state"][-1]
    gripper_dist = tf.cast(gripper_dist > 0.04, tf.bool)
    gripper_dist = tf.cond(
        gripper_dist,
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states[:7], gripper_dist], axis = 0), image

def process_step_austin_mutex(step):
    joint_states = step['observation']['state'][:7] # franka 
    gripper_dist = step["observation"]["state"][7:8]
    gripper_dist = tf.cast(gripper_dist > 0.05, tf.bool)
    gripper_dist = tf.cond(
        gripper_dist,
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states[:7], gripper_dist], axis = 0), image

def process_step_viola(step):
    joint_states = step['observation']['joint_states']  # franka
    gripper_dist = step['observation']['gripper_states'][0]  
    gripper_dist = tf.cast(gripper_dist > 0.04, tf.bool)
    gripper_dist = tf.cond(
        gripper_dist,
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32),
    )
    image = step['observation']['agentview_rgb']
    return tf.concat([joint_states, gripper_dist], axis=0), image

def process_step_taco_play(step):
    robot_obs = step['observation']['robot_obs']  # franka
    joint_states = robot_obs[7:14]  
    gripper_dist = robot_obs[2]       
    gripper_dist = tf.cast(gripper_dist > 0.04, tf.bool)
    gripper_dist = tf.cond(
        gripper_dist,
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32),
    )
    image = step['observation']['rgb_static'] #swap rgb_static with rgb_gripper if want eye-in-hand camera
    return tf.concat([joint_states, gripper_dist], axis=0), image

def process_step_iamlab_cmu(step):
    state = step['observation']['state']  
    joint_states = state[:7]                
    gripper_dist = state[-1]             
    gripper_dist = tf.cast(gripper_dist > 0.04, tf.bool)
    gripper_dist = tf.cond(
        gripper_dist,
        lambda: tf.constant([1], dtype=tf.float32),
        lambda: tf.constant([0], dtype=tf.float32),
    )
    image = step['observation']['image']  
    return tf.concat([joint_states, gripper_dist], axis=0), image

def process_step_bridge(step):
    joint_states = step['observation']['state']
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states, joint_states[-1:]], axis = 0), image

def process_step_furniture_bench(step):
    # TODO: Place holder, need to re-implement
    joint_states = step['observation']['state']
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([joint_states, joint_states[-1:]], axis = 0), image


    