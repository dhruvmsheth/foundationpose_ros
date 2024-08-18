## instructions to run the ros demo:

To run launch file with streaming the output from the folder where the files are stored and foundationpose reading and processing the input
```bash
ros2 launch fp_ros fp_launch.py mesh_file:=src/fp_ros/fp_ros/data/demo_data/unity_sim_data/model/model.obj debug_dir:=src/fp_ros/fp_ros/data/demo_data/unity_sim_data/debug output_dir:=src/fp_ros/fp_ros/data/demo_data/unity_sim_data/output
```

To only run the stream:
```bash
ros2 run fp_ros data_pub
```

To only receive the data from the stream and process using foundationpose (ideal when deploying)
```bash
ros2 run fp_ros data_sub \
--ros-args \
-p mesh_file:=src/fp_ros/fp_ros/data/demo_data/unity_sim_data/model/model.obj \
-p debug_dir:=src/fp_ros/fp_ros/data/demo_data/unity_sim_data/debug \
-p output_dir:=src/fp_ros/fp_ros/data/demo_data/unity_sim_data/output
```
