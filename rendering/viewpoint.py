viewpoints = {
    "toto": {
        "lookat": [ 0.17914434,  0.38201439, -0.05828703],
        "distance": 1.4036129995497302,
        "azimuth": 122.65178571428567,
        "elevation": -28.857142857142883,
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "lookat": [ 0.0468699,  -0.11639435,  0.60702954],
        "distance": 0.4460256912283013,
        "azimuth": 29.454542284691648,
        "elevation": -11.94610988080058,
    },
    "berkeley_autolab_ur5": {
        "lookat": [0.46993696, 0.01620119, 0.50209941],
        "distance": 0.05845199866808603,
        "azimuth": -130.7622432020718,
        "elevation": -38.62050826243951,
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "lookat": [0.35119472, 0.17971953, 0.3853106 ],
        "distance": 1.0161258371793336,
        "azimuth": 90.24973347547972,
        "elevation": -0.13326226012797324,
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "lookat": [0.14865648, 0.06855709, 0.24588401],
        "distance": 0.9889362379730651,
        "azimuth": 177.45885733935935,
        "elevation": -35.487178227549755,
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "lookat": [0.17993816, 0.14028552, 0.22114517],
        "distance": 1.0722252399485346,
        "azimuth": 172.63742876793074,
        "elevation": -32.272892513263926,
    },
    "asu_table_top_converted_externally_to_rlds": {
        "lookat": [ 0.01195814,  0.14997752, -0.13081005],
        "distance": 2.696607306933276,
        "azimuth": -88.89567227736549,
        "elevation": -45.87480888957054,
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "lookat": [ 0.40156015,  0.11487563, -0.19378739],
        "distance": 0.8425373669254637,
        "azimuth": 143.2476649044909,
        "elevation": -38.72600699107954,
    },
    "cmu_play_fusion": {
        "lookat": [0.40401313, 0.08973536, -0.21229478],
        "distance": 0.9010292280980254,
        "azimuth": 133.6127567979282,
        "elevation": -38.99550826243951,
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "lookat": [0.43951549, 0.06844383, -0.21877015],
        "distance": 1.2360590629578143,
        "azimuth": 179.48559042011183,
        "elevation": -63.35311958405548,
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "lookat": [ 0.56822061, -0.1211217,  -0.00574492],
        "distance": 0.7048818324016086,
        "azimuth": 111.06083894803362,
        "elevation": -49.36992403626616,
    },
    "utaustin_mutex": {
        "lookat": [ 0.56659673, -0.02479662,  0.11465414],
        "distance": 1.060807009953462,
        "azimuth": -32.37666105196638,
        "elevation": -29.682424036266163,
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "lookat": [ 0.01085955,  0.09080973, -0.17937824],
        "distance": 0.9828423166021072,
        "azimuth": 168.91798180517654,
        "elevation": -34.369924036266156,
    },
    "bridge": {
        "lookat": [0.31885983952923946, 0.01205641161577117, -0.004560306191141487],
        "distance": 0.3685180587602648,
        "azimuth": 25.824928767930743,
        "elevation": -46.335392513263926,
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
        "lookat": [0.23992831, 0.32660123, 0.40124151],
        "distance": 1.2507610729372296,
        "azimuth": -100.73757123206926,
        "elevation": -32.272892513263926,
    }
}

for name, data in viewpoints.items():
    # name will be the key, e.g., "toto" or "nyu_franka_play_dataset_converted_externally_to_rlds"
    # data will be the dict containing "lookat", "distance", "azimuth", and "elevation"
    azimuth_value = data["azimuth"]
    elevation_value = data["elevation"]
    print(name)
    print("\"yaw\":", 180 + azimuth_value)
    print("\"roll\":", 90 + elevation_value)