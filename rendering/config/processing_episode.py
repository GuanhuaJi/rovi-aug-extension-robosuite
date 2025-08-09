import os
import tensorflow as tf
import numpy as np
import math

def _expand_flag(flag_bool):
    """bool (T,) → float32 (T,1)"""
    return tf.expand_dims(tf.cast(flag_bool, tf.float32), -1)


def process_episode_toto(episode):
    stp   = episode["steps"]
    joint = tf.cast(stp["observation"]["state"], tf.float32)
    joint = joint + tf.constant(
        [0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2, math.pi / 4],
        dtype=tf.float32,
    )
    flag  = _expand_flag(stp["action"]["open_gripper"] > 0.5)
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, flag], axis=-1), imgs

def process_episode_nyu_franka(episode):
    stp = episode["steps"]
    joint = stp["observation"]["state"][:, :7]                      # (T,7)
    flag = _expand_flag(stp["action"][:, 13] > 0)                  # (T,1)
    imgs = stp["observation"]["image"]                             # (T,H,W,3)
    return tf.concat([joint, flag], axis=-1), imgs                 # (T,8), imgs


def process_episode_autolab_ur5(episode):
    stp   = episode["steps"]
    rbot  = stp["observation"]["robot_state"]
    joint = tf.cast(rbot[:, :6], tf.float32)
    joint += tf.constant([0, 0, 0, 0, 0, math.pi / 2], dtype=tf.float32)
    flag  = _expand_flag(rbot[:, -2] > 0.5)
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, flag], axis=-1), imgs


def process_episode_ucsd_kitchen_rlds(episode):
    stp   = episode["steps"]
    joint = stp["observation"]["state"][:, :7]                     # (T,7)
    flag  = _expand_flag(stp["action"][:, 6] < 0.5)                # (T,1)
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, flag], axis=-1), imgs                 # (T,8)


def process_episode_utokyo_pick_and_place(episode):
    stp   = episode["steps"]
    joint = stp["observation"]["joint_state"][:, :7]               # (T,7)
    flag  = _expand_flag(stp["action"][:, -1] > 0.5)               # (T,1)
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, flag], axis=-1), imgs


def process_episode_asu_table_top_rlds(episode):
    stp   = episode["steps"]
    state = tf.cast(stp["observation"]["state"], tf.float32)
    joint = state[:, :6]
    sign   = tf.constant([1., 1., -1., 1., 1., 1.], dtype=tf.float32)
    offset = tf.constant([0., math.pi/2, 0., math.pi/2, 0., math.pi/2],
                         dtype=tf.float32)
    joint = joint * sign - offset
    flag  = _expand_flag(state[:, -1] > 0.2)
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, flag], axis=-1), imgs

def process_episode_kaist(episode):
    stp   = episode["steps"]
    joint = stp["observation"]["state"][:, :14:2]                  # (T,7)
    flag  = tf.zeros((tf.shape(joint)[0], 1), tf.float32)          # gripper unused
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, flag], axis=-1), imgs

def process_episode_austin_buds(episode):
    _TOL = 1e-8
    stp   = episode["steps"]
    state = tf.cast(stp["observation"]["state"], tf.float32)
    joint_raw = state[:, :7]
    def _fill_zero_rows(j_np):
        """
        NumPy implementation of your original while-loop, run once per
        episode inside tf.numpy_function.
        """
        j = j_np.copy()
        N = len(j)
        zero_mask = np.all(np.isclose(j, 0.0, atol=_TOL), axis=1)

        if zero_mask.all():
            print("WARNING: all joint_angles rows are zeros; nothing replaced.")
            return j

        i = 0
        while i < N:
            if not zero_mask[i]:
                i += 1
                continue
            start = i
            while i < N and zero_mask[i]:
                i += 1
            end = i

            left  = start - 1
            right = end if end < N else None

            if left < 0 and right is None:
                print("WARNING: joint_angles entirely zero for austin_buds; left unchanged.")
                break
            if left < 0:
                j[start:end] = j[right]
            elif right is None:
                j[start:end] = j[left]
            else:
                gap = right - left
                for k in range(1, gap):
                    alpha = k / gap
                    j[left + k] = (1 - alpha) * j[left] + alpha * j[right]

        print(f"[INFO] austin_buds: filled {zero_mask.sum()} zero rows via interpolation / copying.")
        return j.astype(np.float32)

    joint_fixed = tf.numpy_function(_fill_zero_rows, [joint_raw], tf.float32)
    joint_fixed.set_shape([None, 7])
    flag = _expand_flag(state[:, 7] > 0.04)
    imgs = stp["observation"]["image"]

    return tf.concat([joint_fixed, flag], axis=-1), imgs


def process_episode_austin_sailor(episode):
    stp   = episode["steps"]
    joint = stp["observation"]["state_joint"]                      # (T,7)
    flag  = _expand_flag(stp["observation"]["state"][:, -1] > 0.04)
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, flag], axis=-1), imgs


def process_episode_austin_mutex(episode):
    stp   = episode["steps"]
    state = stp["observation"]["state"]
    joint = state[:, :7]
    flag  = _expand_flag(state[:, 7] > 0.05)
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, flag], axis=-1), imgs

def process_episode_viola(episode):
    _TOL = 1e-8
    stp   = episode["steps"]
    joint_raw = tf.cast(stp["observation"]["joint_states"], tf.float32)
    non_zero      = tf.reduce_any(tf.abs(joint_raw) > _TOL, axis=-1)   # (T,)
    any_non_zero  = tf.reduce_any(non_zero)

    def _replace():
        first_idx  = tf.argmax(tf.cast(non_zero, tf.int32), output_type=tf.int32)
        first_row  = joint_raw[first_idx]                              # (7,)
        leading    = tf.repeat(first_row[tf.newaxis, :], first_idx, axis=0)
        tf.print("WARNING: first", first_idx,
                 "rows were zeros; copied row", first_idx, "into them.")
        return tf.concat([leading, joint_raw[first_idx:]], axis=0)

    def _no_replace():
        tf.print("WARNING: all joint_angles rows are zeros; nothing replaced.")
        return joint_raw
    joint = tf.cond(any_non_zero, _replace, _no_replace)               # (T,7)
    flag  = _expand_flag(stp["observation"]["gripper_states"][:, 0] > 0.04)
    imgs  = stp["observation"]["agentview_rgb"]
    return tf.concat([joint, flag], axis=-1), imgs


def process_episode_taco_play(episode):
    stp   = episode["steps"]
    robo  = stp["observation"]["robot_obs"]
    joint = robo[:, 7:14]                                          # (T,7)
    flag  = _expand_flag(robo[:, 2] > 0.04)
    imgs  = stp["observation"]["rgb_static"]  # use rgb_gripper for eye-in-hand
    return tf.concat([joint, flag], axis=-1), imgs


def process_episode_iamlab_cmu(episode):
    stp   = episode["steps"]
    state = stp["observation"]["state"]
    joint = state[:, :7]
    flag  = _expand_flag(state[:, -1] > 0.04)
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, flag], axis=-1), imgs


def process_episode_bridge(episode):
    stp   = episode["steps"]
    joint = stp["observation"]["state"]
    extra = joint[:, -1:]                                          # (T,1)
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, extra], axis=-1), imgs

def process_episode_furniture_bench(episode):
    stp   = episode["steps"]
    joint = stp["observation"]["state"]
    extra = joint[:, -1:]
    imgs  = stp["observation"]["image"]
    return tf.concat([joint, extra], axis=-1), imgs