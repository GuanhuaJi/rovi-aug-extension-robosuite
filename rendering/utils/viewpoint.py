# viewpoints = {
#     "toto": {
#         "lookat": [ 0.17914434,  0.38201439, -0.05828703],
#         "distance": 1.4036129995497302,
#         "azimuth": 122.65178571428567, # 28
#         "elevation": -28.857142857142883, # 67
#     },
#     "nyu_franka_play_dataset_converted_externally_to_rlds": {
#         "lookat": [ 0.0468699,  -0.11639435,  0.60702954],
#         "distance": 0.4460256912283013,
#         "azimuth": 29.454542284691648, # -57.0
#         "elevation": -11.94610988080058, # 78
#     },
#     "berkeley_autolab_ur5": {
#         "lookat": [0.46993696, 0.01620119, 0.50209941],
#         "distance": 0.05845199866808603,
#         "azimuth": -130.7622432020718, # 49.2377567979282
#         "elevation": -38.62050826243951, # 51.37949173756049
#     },
#     "ucsd_kitchen_dataset_converted_externally_to_rlds": {
#         "lookat": [0.35119472, 0.17971953, 0.3853106 ],
#         "distance": 1.0161258371793336,
#         "azimuth": 90.24973347547972,
#         "elevation": -0.13326226012797324,
#     },
#     "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
#         "lookat": [0.14865648, 0.06855709, 0.24588401],
#         "distance": 0.9889362379730651,
#         "azimuth": 177.45885733935935,
#         "elevation": -35.487178227549755,
#     },
#     "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
#         "lookat": [0.17993816, 0.14028552, 0.22114517],
#         "distance": 1.0722252399485346,
#         "azimuth": 172.63742876793074,
#         "elevation": -32.272892513263926,
#     },
#     "asu_table_top_converted_externally_to_rlds": {
#         "lookat": [ 0.01195814,  0.14997752, -0.13081005],
#         "distance": 2.696607306933276,
#         "azimuth": -88.89567227736549,
#         "elevation": -45.87480888957054, # 52
#     },
#     "kaist_nonprehensile_converted_externally_to_rlds": {
#         "lookat": [ 0.40156015,  0.11487563, -0.19378739],
#         "distance": 0.8425373669254637,
#         "azimuth": 143.2476649044909,
#         "elevation": -38.72600699107954, #51.3
#     },
#     "cmu_play_fusion": {
#         "lookat": [0.40401313, 0.08973536, -0.21229478],
#         "distance": 0.9010292280980254,
#         "azimuth": 133.6127567979282,
#         "elevation": -38.99550826243951,
#     },
#     "austin_buds_dataset_converted_externally_to_rlds": {
#         "lookat": [0.43951549, 0.06844383, -0.21877015],
#         "distance": 1.2360590629578143,
#         "azimuth": 179.48559042011183, # 89
#         "elevation": -63.35311958405548, # 29
#     },
#     "austin_sailor_dataset_converted_externally_to_rlds": {
#         "lookat": [ 0.56822061, -0.1211217,  -0.00574492],
#         "distance": 0.7048818324016086,
#         "azimuth": 111.06083894803362, # 22
#         "elevation": -49.36992403626616, # 35.0
#     },
#     "utaustin_mutex": {
#         "lookat": [ 0.56659673, -0.02479662,  0.11465414],
#         "distance": 1.060807009953462,
#         "azimuth": -32.37666105196638, # -130
#         "elevation": -29.682424036266163, # 66
#     },
#     "austin_sirius_dataset_converted_externally_to_rlds": {
#         "lookat": [ 0.01085955,  0.09080973, -0.17937824],
#         "distance": 0.9828423166021072,
#         "azimuth": 168.91798180517654,
#         "elevation": -34.369924036266156, 
#     },
#     "bridge": {
#         "lookat": [0.31885983952923946, 0.01205641161577117, -0.004560306191141487],
#         "distance": 0.3685180587602648,
#         "azimuth": 25.824928767930743,
#         "elevation": -46.335392513263926,
#     },
#     "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
#         "lookat": [0.23992831, 0.32660123, 0.40124151],
#         "distance": 1.2507610729372296,
#         "azimuth": -100.73757123206926,
#         "elevation": -32.272892513263926,
#     }
# }

viewpoints = {
    "toto": {
        #average of bottom 2 parameters
        "lookat": [0.360729295, 0.106302025, 0.13910922],
        "distance": 1.03,
        "azimuth": 119.8125,
        "elevation": -18.59375,
        # "lookat": [0.36068405, 0.10638337, 0.13733144],
        # "distance": 1.03,
        # "azimuth": 119.625,
        # "elevation": -19.8125,

        # "lookat": [0.36077454, 0.10622068, 0.140887],
        # "distance": 1.03,
        # "azimuth": 120.0,
        # "elevation": -17.375,
        "process_function" : process_step_toto
    },

    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "lookat": [ 0.0468699,  -0.11639435,  0.60702954],
        "distance": 0.4460256912283013,
        "azimuth": 29.454542284691648, #-57.0
        "elevation": -11.94610988080058, #78
        "camera_fov": 45,
        "process_function" : process_step_nyu
    },
    "berkeley_autolab_ur5": {
        "lookat": [0.24398564, -0.23442528,  0.25454247],
        "distance": 0.361531964776284,
        "azimuth": -129.6372432020718,
        "elevation": -42.125, #51
        "camera_fov": 57.82240163683314,
        "process_function" : process_step_berkeley_ur5
    },
    #can also use this, but I think it's better to use the one above (more accurate in my opinion)
    # "berkeley_autolab_ur5": {
    #     "lookat": [0.24503305, -0.23506916,  0.22667052],
    #     "distance": 0.37715926814216433,
    #     "azimuth": -129.0747432020718,
    #     "elevation":  -37.25,
    #     "camera_fov": 57.82240163683314,
    #     "process_function" : process_step_berkeley_ur5
    # },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "lookat": [0.35624648, 0.18027023, 0.26440381 ],
        "distance": 1.0102697170162116,
        "azimuth": 91.74973347547972,
        "elevation": -3.5082622601279727,
        "camera_fov": 45,
        "process_function" : process_step_ucsd_kitchen
    },
    #this one is a more genralized one that's taken the average of the 15 and 45 and tilted a little
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "lookat": [0.16423974, 0.0605987,  0.13352152],
        "distance": 0.9653874337601709,
        "azimuth": 174.6463573393588,
        "elevation": -34.48271394183545,
        "camera_fov": 45,
        "process_function": process_step_utokyo_xarm_pick_place
    },
    #this one works well for frame 15
    # "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
    #     "lookat": [0.15925299, 0.06681606, 0.12932121],
    #     "distance": 0.9780654869184446,
    #     "azimuth": 176.6552859107874,
    #     "elevation": -31.067535370406866,
    #     "process_function" : process_step_utokyo_xarm_pick_place
    # },
    # this one works well for frame 45
    # "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
    #     "lookat": [0.16922649, 0.05438134, 0.13772182],
    #     "distance": 0.9527093806018971,
    #     "azimuth": 171.03028591078734,
    #     "elevation": -31.469321084692595,
    #     "process_function" : process_step_utokyo_xarm_pick_place
    # },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "lookat": [0.16922649, 0.05438134, 0.13772182],
        "distance": 0.9527093806018971,
        "azimuth": 171.03028591078734,
        "elevation": -33.47824965612117,
        "camera_fov": 45,
        "process_function" : process_step_ucsd_pick_and_place
    },
    "asu_table_top_converted_externally_to_rlds": {
        "lookat": [0.01263838,  0.09175601, -0.17026123],
        "distance": 2.7237191354279924,
        "azimuth": -85.27960084879423,
        "elevation":  -40.65159460385623,
        "camera_fov": 45,
        "process_function" : process_step_asu_table_top
    }, #done

    "kaist_nonprehensile_converted_externally_to_rlds": {
        "lookat": [0.40113128,  0.1131025,  -0.20421406],
        "distance": 0.9116127545512921,
        "azimuth": 139.6851649044909,
        "elevation": -37.03850699107953,
        "camera_fov": 43.0, 
        "process_function" : process_step_kaist_nonprehensile
    },

    #can also use this, but I think it's better to use the one above (more accurate in my opinion)
    # "kaist_nonprehensile_converted_externally_to_rlds": {
    #     "lookat": [0.39967922,  0.11139119, -0.20571029],
    #     "distance": 0.9116127545512921,
    #     "azimuth": 140.0601649044909,
    #     "elevation": -36.10100699107953,
    #     "camera_fov": 43.0, 
    #     "process_function" : process_step_kaist_nonprehensile
    # },

    "cmu_play_fusion": {
        "lookat": [0.40401313, 0.08973536, -0.21229478],
        "distance": 0.9010292280980254,
        "azimuth": 133.6127567979282,
        "elevation": -38.99550826243951,
        "camera_fov": 45,
        "process_function" : process_step_cmu_play_fusion
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "lookat": [0.43482858,  0.08394756, -0.18341207],
        "distance": 1.1236742324339846,
        "azimuth": 179.10815957988817,
        "elevation": -60.54061958405549,
        "camera_fov": 50,
        "process_function" : process_step_austin_buds
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "lookat": [0.35071924,  0.39197068, -0.66741147],
        "distance": 1.570611132488425,
        "azimuth": 113.87333894803362,
        "elevation": -49.36992403626616,
        "camera_fov": 46,
        "process_function" : process_step_austin_sailor
    },
    "utaustin_mutex": {
        "lookat": [0.43803405, 0.04173897, 0.19695073],
        "distance": 0.6745413646576816,
        "azimuth": -37.29853605196638,
        "elevation": -32.75,
        "camera_fov": 60,
        "process_function" : process_step_austin_mutex
    },

    #Does not work
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "lookat": [0.49165285, -0.00759665,  0.16697217],
        "distance": 0.4158978638973598,
        "azimuth": 164.63226751946206,
        "elevation": -37.57142857142855,
        "camera_fov": 30,
        "process_function" : process_step_austin_sirius
    },
    "bridge": {
        "lookat": [0.31885983952923946, 0.01205641161577117, -0.004560306191141487],
        "distance": 0.3685180587602648,
        "azimuth": 25.824928767930743,
        "elevation": -46.335392513263926,
        "camera_fov": 45,
        "process_function" : process_step_bridge
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
        "lookat": [0.23992831, 0.32660123, 0.40124151],
        "distance": 1.2507610729372296,
        "azimuth": -100.73757123206926,
        "elevation": -32.272892513263926,
        "camera_fov": 45,
        "process_function" : process_step_stanford_hydra
    }, 
    # "viola": {
    #     #scene 1
    #     "lookat": [ 0.42379717,  0.0389163,  -0.25703527],  
    #     "distance": 1.3490181765881022,
    #     "azimuth": -179.59821428571428, # 90.0
    #     "elevation": -46.453571285714276, # 38.0
    #     "camera_fov": 36.0,
    #     "process_function": process_step_viola
    # }, 
    "viola": {
        #scene 2
        "lookat": [0.52363341, 0.15771185, 0.05883797],  
        "distance": 1.06776311996394,
        "azimuth": 120.93749999999997, # 35.0
        "elevation": -32.79285699999997, # 56.0
        "camera_fov": 36.0,
        "process_function": process_step_viola
    }, 




    # "taco_play": {
    #option 1
    #     "lookat": [ 0.36733439, -0.06561337,  0.37078567],       #trajectory 45 - [0.35715356, -0.06449276, 0.36707649]  
    #     "distance": 1.278414928422551,
    #     "azimuth": 37.799999999999976,
    #     "elevation": -38.999999999999986,
    #     # "camera_fov": 40,
    #     "process_function": process_step_taco_play
    # },
    # "taco_play": {
    #option 2
    #     "lookat": [ 0.3934845,  -0.09778859,  0.34475987],       #trajectory 45 - [0.35715356, -0.06449276, 0.36707649]  
    #     "distance": 1.3577269466017963,
    #     "azimuth": 38.99999999999998,
    #     "elevation": -43.199999999999996,
    #     # "camera_fov": 40,
    #     "process_function": process_step_taco_play
    # },

    # "taco_play": {
    #     #option 3
    #     "lookat": [0.35270713, 0.00521408, 0.30643632],
    #     "distance": 1.3945714662711075,
    #     "azimuth": 48.00000000000002,
    #     "elevation": -41.600000000000044,
    #     # "camera_fov": 40,
    #     "process_function": process_step_taco_play
    # },
    "taco_play": {
        #combined
        "lookat": [0.37802107, -0.040776484, 0.283334922],
        "distance": 1.3945714662711075,
        "azimuth": 39.72, # 45
        "elevation": -40.88, # 50
        # "camera_fov": 40,
        "process_function": process_step_taco_play
    },




    # "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
    #     #top view option two
    #     "lookat": [ 0.6796343,  -0.0078609,   0.65170916],       
    #     "distance": 0.1,
    #     "azimuth": 179.25,
    #     "elevation": -84.5,
    #     # "camera_fov": 50,
    #     "process_function": process_step_iamlab_cmu_pickup_insert
    # },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        #top view best
        "lookat": [0.6798035,  0.00979163, 0.78616425],       
        "distance": 0.008682539698261531,
        "azimuth": -178.25,
        "elevation": -85.75,
        # "camera_fov": 38,
        "process_function": process_step_iamlab_cmu_pickup_insert
    },

    # "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
    #     #side view
    #     "lookat": [0.55570769, 0.02037676, 0.00339291],       
    #     "distance": 0.6621527150355037,
    #     "azimuth": -90.5,
    #     "elevation": -37.75,
    #     # "camera_fov": 50,
    #     "process_function": process_step_iamlab_cmu_pickup_insert
    # }
}


import numpy as np
from scipy.spatial.transform import Rotation as R

def spherical_to_cartesian(dist, az_deg, el_deg):
    """Return XYZ offset for given distance / azimuth / elevation."""
    az, el = np.deg2rad([az_deg, el_deg])
    x = dist * np.cos(el) * np.cos(az)
    y = dist * np.cos(el) * np.sin(az)
    z = dist * np.sin(el)
    return np.array([x, y, z])

def camera_quat(az_deg, el_deg):
    """Quaternion (xyzw) that matches MuJoCo free-camera rules."""
    yaw   = np.deg2rad(180 + az_deg)
    roll  = np.deg2rad(90  + el_deg)
    pitch = 0.0
    return R.from_euler('xyz', [roll, pitch, yaw]).as_quat()  # xyzw

results = {}

for name, vp in viewpoints.items():
    lookat   = np.asarray(vp["lookat"])
    offset   = spherical_to_cartesian(vp["distance"],
                                      vp["azimuth"],
                                      vp["elevation"])
    cam_pos  = lookat + offset
    cam_quat = camera_quat(vp["azimuth"], vp["elevation"])

    results[name] = {
        "position": cam_pos.tolist(),      # XYZ
        "quat_xyzw": cam_quat.tolist(),    # (x, y, z, w)
    }

# ---------------------------------------------------------------------
# 3.  pretty-print  ----------------------------------------------------
for name, pose in results.items():
    print(f"{name}:")
    print("  pos :", np.round(pose['position'], 6).tolist())
    print("  quat:", np.round(pose['quat_xyzw'], 6).tolist())
    print()