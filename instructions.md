## Steps required to train the 3D representation of any custom object:

First, transfer the data to the docker container. 

To train on custom data, the data format should be as follows:

- main folder
    - rgb: contains 16 images of the object
    - depth: contains the corresponding depth for all the images
    - mask: contains the mask for which the obj has to be extracted
    - cam_in_ob: contains the camera orientation as a 4x4 matrix without any ',' and with the same filename as rgb and depth
    - K.txt: contains the intrinsics of the camera used for collecting the data


Then, run 
```
python3 bundlesdf/run_nerf.py --ref_view_dir /home/FoundationPose/views/ref_views_16 (path to ref folder) --dataset ycbv
```

Check the `model` folder for the .obj output which would be requied to run the main model. Then, transfer the data to test on into docker container for testing. 
Then run:

```
python3 run_demo_store.py --test_scene_dir demo_data/outputs (the output you collected) --mesh_file demo_data/outputs/mesh/model.obj (the .obj you obtained with texture files in the same folder)
```


