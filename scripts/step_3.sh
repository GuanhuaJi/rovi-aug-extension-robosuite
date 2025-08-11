python ../r2r-robosuite/export_source_robot_states_sim.py --robot_dataset=can \
         --hdf5_path=<PATH TO CAN HDF5> \
            --workers=10 --chunksize=40

python ../r2r-robosuite/export_source_robot_states_sim.py --robot_dataset=lift \
         --hdf5_path=<PATH TO LIFT HDF5> \
            --workers=10 --chunksize=40

python ../r2r-robosuite/export_source_robot_states_sim.py --robot_dataset=square \
         --hdf5_path=<PATH TO SQUARE HDF5> \
            --workers=10 --chunksize=40

python ../r2r-robosuite/export_source_robot_states_sim.py --robot_dataset=stack \
         --hdf5_path=<PATH TO STACK HDF5> \
            --workers=10 --chunksize=40

python ../r2r-robosuite/export_source_robot_states_sim.py --robot_dataset=two_piece \
         --hdf5_path=<PATH TO TWO_PIECE HDF5> \
            --workers=10 --chunksize=40