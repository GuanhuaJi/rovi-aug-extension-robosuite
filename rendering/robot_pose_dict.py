import numpy as np

ROBOT_POSE_DICT = {
    "austin_buds": {
        "Panda": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
    },
    "austin_sailor": {
        "Panda": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
    },
    "autolab_ur5": {
        "Panda": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
    },
    "austin_mutex": {
        "Panda": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
    },   
    "nyu_franka": {
        "Panda": {
            "safe_angle": [-0.6102706193923950195, -1.455744981765747070, 1.501405358314514160, -2.240571022033691406, -0.2229462265968322754, 2.963621616363525391, -0.5898305177688598633],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [-0.095583,  0.246849,  0.210468, -1.817155,  6.216322,  1.064266,  6.393936],
            "displacement": np.array([-0.3, 0, 0.1])
        },
        "Sawyer": {
            "safe_angle": [0.2553876936435699463, -0.03010351583361625671, -0.9372422099113464355, 1.432788133621215820, 1.421311497688293457, 0.9351797103881835938, -3.228875875473022461],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [-3.090315, 1.409086,  1.500532, 1.821214, -4.4519,    4.718327,  1.770046],
            "displacement": np.array([0, 0, 0])
        },
    },
    "kaist": {
        "Panda": {
            "safe_angle": [-0.6102706193923950195, -1.455744981765747070, 1.501405358314514160, -2.240571022033691406, -0.2229462265968322754, 2.963621616363525391, -0.5898305177688598633],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [-0.095583,  0.246849,  0.210468, -1.817155,  6.216322,  1.064266,  6.393936],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [0.2553876936435699463, -0.03010351583361625671, -0.9372422099113464355, 1.432788133621215820, 1.421311497688293457, 0.9351797103881835938, -3.228875875473022461],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [-3.090315, 1.409086,  1.500532, 1.821214, -4.4519,    4.718327,  1.770046],
            "displacement": np.array([0, 0, 0])
        },
    },
    "toto": {
        "Panda": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
    },
    "asu_table_top_rlds": {
        "Panda": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
    },
    "taco_play": {
        "Panda": {
            "safe_angle": [-0.6102706193923950195, -1.455744981765747070, 1.501405358314514160, -2.240571022033691406, -0.2229462265968322754, 2.963621616363525391, -0.5898305177688598633],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [-0.095583,  0.246849,  0.210468, -1.817155,  6.216322,  1.064266,  6.393936],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [0.2553876936435699463, -0.03010351583361625671, -0.9372422099113464355, 1.432788133621215820, 1.421311497688293457, 0.9351797103881835938, -3.228875875473022461],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [-3.090315, 1.409086,  1.500532, 1.821214, -4.4519,    4.718327,  1.770046],
            "displacement": np.array([0, 0, 0])
        },
    },
    "furniture_bench": {
        "Panda": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
    },
    "viola": {
        "Panda": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [],
            "displacement": np.array([0, 0, 0])
        },
    },
    "iamlab_cmu": {
        "Panda": {
            "safe_angle": [-0.6102706193923950195, -1.455744981765747070, 1.501405358314514160, -2.240571022033691406, -0.2229462265968322754, 2.963621616363525391, -0.5898305177688598633],
            "displacement": np.array([0, 0, 0])
        },
        "IIWA": {
            "safe_angle": [-0.095583,  0.246849,  0.210468, -1.817155,  6.216322,  1.064266,  6.393936],
            "displacement": np.array([0, 0, 0])
        },
        "Sawyer": {
            "safe_angle": [0.2553876936435699463, -0.03010351583361625671, -0.9372422099113464355, 1.432788133621215820, 1.421311497688293457, 0.9351797103881835938, -3.228875875473022461],
            "displacement": np.array([0, 0, 0])
        },
        "Jaco": {
            "safe_angle": [-3.090315, 1.409086,  1.500532, 1.821214, -4.4519,    4.718327,  1.770046],
            "displacement": np.array([0, 0, 0])
        },
    }
}