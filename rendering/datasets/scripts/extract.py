import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Define the path for saving the episodes
SAVE_PATH = 'dataset'

'''
    'fractal20220817_data',
    'kuka',
    'bridge',
    'taco_play',
    'jaco_play',
    'berkeley_cable_routing',
    'roboturk',
    'nyu_door_opening_surprising_effectiveness',
    'viola',
    'berkeley_autolab_ur5',
    'toto',
    'language_table',
    'columbia_cairlab_pusht_real',
    'stanford_kuka_multimodal_dataset_converted_externally_to_rlds',
    'nyu_rot_dataset_converted_externally_to_rlds',
    'stanford_hydra_dataset_converted_externally_to_rlds',
    'austin_buds_dataset_converted_externally_to_rlds',
    'nyu_franka_play_dataset_converted_externally_to_rlds',
    'maniskill_dataset_converted_externally_to_rlds',
    'cmu_franka_exploration_dataset_converted_externally_to_rlds',
    'ucsd_kitchen_dataset_converted_externally_to_rlds',
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
    'austin_sailor_dataset_converted_externally_to_rlds',
    'austin_sirius_dataset_converted_externally_to_rlds',
    'bc_z',
    'usc_cloth_sim_converted_externally_to_rlds',
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds',
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds',
'''


'''
'utokyo_saytap_converted_externally_to_rlds', 20
'''
DATASETS = [
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds',
    'furniture_bench_dataset_converted_externally_to_rlds',
    'fractal20220817_data',
    'kuka',
    'bridge',
    'taco_play',
    'jaco_play',
    'berkeley_cable_routing',
    'roboturk',
    'nyu_door_opening_surprising_effectiveness',
    'viola',
    'berkeley_autolab_ur5',
    'toto',
    'language_table',
    'columbia_cairlab_pusht_real',
    'stanford_kuka_multimodal_dataset_converted_externally_to_rlds',
    'nyu_rot_dataset_converted_externally_to_rlds',
    'stanford_hydra_dataset_converted_externally_to_rlds',
    'austin_buds_dataset_converted_externally_to_rlds',
    'nyu_franka_play_dataset_converted_externally_to_rlds',
    'maniskill_dataset_converted_externally_to_rlds',
    'cmu_franka_exploration_dataset_converted_externally_to_rlds',
    'ucsd_kitchen_dataset_converted_externally_to_rlds',
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
    'austin_sailor_dataset_converted_externally_to_rlds',
    'austin_sirius_dataset_converted_externally_to_rlds',
    'bc_z',
    'usc_cloth_sim_converted_externally_to_rlds',
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds',
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds',
    "fmb",
    "io_ai_tech",
    "mimic_play",
    "aloha_mobile",
    "robo_set",
    "tidybot",
    "vima_converted_externally_to_rlds",
    "spoc",
    "plex_robosuite",
    "stanford_mask_vit_converted_externally_to_rlds",
    "kaist_nonprehensile_converted_externally_to_rlds",
    "asu_table_top_converted_externally_to_rlds"
]

# Function to map dataset names to the correct paths
def dataset2path(dataset_name):
    if dataset_name == "fmb":
        return f'gs://gresearch/robotics/{dataset_name}/0.1.1'
    return f'gs://gresearch/robotics/{dataset_name}/0.1.0'

display_key = 'state'

# Load the dataset
for dataset in DATASETS:
    print(dataset)
    try:
        # Build the dataset from the GCS path
        b = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
        print(b.info.features)

        continue
        # Check if the desired display_key exists
        if display_key not in b.info.features['steps']['observation']:
            print("No Image")
            continue


        # Get the total number of episodes in the dataset
        total_episodes = b.info.splits['train'].num_examples
        
        # If there are fewer than 20 episodes, take all; otherwise, take the first 20
        split = 'train[:20]' if total_episodes >= 20 else 'train'
        ds = b.as_dataset(split=split)
        
        for episode_num, episode in enumerate(ds):
            print(episode)
    
    except Exception as e:
        print(f"Error processing dataset {dataset}: {e}")

'''
# Load the dataset
for dataset in DATASETS:
    print(dataset)
    try:
        # Build the dataset from the GCS path
        b = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
        
        # Check if the desired display_key exists
        if display_key not in b.info.features['steps']['observation']:
            print("No Image")
            continue

        # Get the total number of episodes in the dataset
        total_episodes = b.info.splits['train'].num_examples
        
        # If there are fewer than 20 episodes, take all; otherwise, take the first 20
        split = 'train[:20]' if total_episodes >= 20 else 'train'
        ds = b.as_dataset(split=split)
        
        for episode_num, episode in enumerate(ds):
            # Extract state information from steps
            state = np.stack([step['observation']['state'].numpy() for step in episode['steps']])
            
            # Create the output folder dynamically
            folder_path = f'./states/{dataset}/'
            os.makedirs(folder_path, exist_ok=True)
            
            # Save the extracted data
            np.savetxt(folder_path + f'ee_states_{episode_num}.txt', state[:, -16:])
            np.savetxt(folder_path + f'joint_states_{episode_num}.txt', state[:, :7])
    
    except Exception as e:
        print(f"Error processing dataset {dataset}: {e}")
'''