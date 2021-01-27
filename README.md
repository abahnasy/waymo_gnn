# Waymo Tracking

### TODO
- [ ] requirements.txt
- [ ] Docker file
- [ ] List prerequisites
- [ ] header comments


### Steps

* TODO: GCloud access and gcp cmd tools
* TODO: reuqired installations !
* build cuda layers of dcn and iou3d_nms `bash setup.sh`
* alias to data folder
* run the downloader to download in the alias folder


### CMD Lines

* alias data folder

* download data

``` python3 download_tfrecords.py --split 'training'  --root_path './data/Waymo' ```

* preprocess data

    * train set 
    ```CUDA_VISIBLE_DEVICES=-1 python3 waymo_dataset/waymo_converter.py --tfrecord_path 'data/Waymo/tfrecord_training/segment-*.tfrecord'  --root_path 'data/Waymo/train/'```

    * validation set 
    ```CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --tfrecord_path 'WAYMO_DATASET_ROOT/tfrecord_validation/segment-*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/val/'```

    * testing set 
    ```CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --tfrecord_path 'WAYMO_DATASET_ROOT/tfrecord_testing/segment-*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/test/'```

* create info files
# train
``` python3 tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=1 ```
# val
```python3 tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=1```


# train cmds
```python3 ./tools/train.py CONFIG_PATH```

# test cmds
```python3 ./tools/dist_test.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3epoch.py --work_dir work_dirs/waymo_centerpoint_voxelnet_3epoch --checkpoint work_dirs/waymo_centerpoint_voxelnet_3epoch/latest.pth --speed_test```

# generate ground truth for validation set
```python3 det3d/datasets/waymo/waymo_common.py --info_path data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl --result_path data/Waymo/ --gt```



# From inside Waymo Folder to evaluate predicitons !
```bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main /home/bahnasy/CenterPoint/work_dirs/waymo_centerpoint_voxelnet_3epoch/detection_pred.bin /home/bahnasy/CenterPoint/data/Waymo/gt_preds.bin```


# Backlog
- [ ] numba jit warning ! 