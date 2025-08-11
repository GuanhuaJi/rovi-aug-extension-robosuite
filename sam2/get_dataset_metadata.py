import tensorflow_datasets as tfds

DATASETS = [
    "toto",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "berkeley_autolab_ur5",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    "kaist_nonprehensile_converted_externally_to_rlds",
    "asu_table_top_converted_externally_to_rlds",
    "austin_buds_dataset_converted_externally_to_rlds",
    "utaustin_mutex",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "bridge",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "viola",
    "taco_play",
    "language_table",
    "fractal20220817_data"
]


def main():
    for ds_name in DATASETS:
        print(f"\n===== {ds_name} =====")

        builder = tfds.builder_from_directory(
            f"gs://gresearch/robotics/{ds_name}/0.1.0/"
        )

        # Pull metadata (and shards if you haven’t cached them yet)
        builder.download_and_prepare()

        # Print split names and episode counts
        for split_name, split_info in builder.info.splits.items():
            # In these RLDS datasets, each TF-Example is one episode
            num_episodes = split_info.num_examples
            print(f"{split_name:<15} : {num_episodes:6,d} episodes")


if __name__ == "__main__":
    main()