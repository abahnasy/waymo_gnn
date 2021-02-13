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

* install open_waymo_dataset library to extract data
```pip install waymo-open-dataset-tf-2-3-0```

### CMD Lines

* alias data folder

* download data

``` python3 download_tfrecords.py --split 'training'  --root_path './data/Waymo' ```

* preprocess data, extract annotaions and point clouds for every frame into piclke file

    * train set 
    ```CUDA_VISIBLE_DEVICES=-1 python3 waymo_dataset/waymo_converter.py --tfrecord_path 'data/Waymo/tfrecord_training/segment-*.tfrecord'  --root_path '.data/Waymo/train/'```

    * validation set 
    ```CUDA_VISIBLE_DEVICES=-1 python3 waymo_dataset/waymo_converter.py --tfrecord_path 'data/Waymo/tfrecord_validation/segment-*.tfrecord'  --root_path '.data/Waymo/val/'```

    * testing set 
    ```CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --tfrecord_path 'WAYMO_DATASET_ROOT/tfrecord_testing/segment-*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/test/'```

* create info files
# train
``` waymo_dataset/create_data.py waymo_data_prep --root_path=./data/Waymo --split train --nsweeps=1 --sub_sampled=1.0 ```
# val
```python3 waymo_dataset/create_data.py waymo_data_prep --root_path=./data/Waymo --split val --nsweeps=1 --sub_sampled=0.1```



# train cmds
```python3 ./tools/train.py CONFIG_PATH```

# test cmds
```python3 ./dist_test.py --work_dir work_dirs/dry_run --checkpoint work_dirs/dry_run/latest.pth --speed_test```

# generate ground truth for validation set
```python3 waymo_open_dataset/waymo_common.py --info_path data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl --result_path data/Waymo/ --gt```



# From inside Waymo Folder to evaluate predicitons !
```bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main /home/bahnasy/waymo_gnn/work_dirs/dry_run/detection_pred.bin /home/bahnasy/waymo_gnn/data/Waymo/gt_preds.bin```


# Launch Tensorboard
```python3 -m tensorboard.main --logdir work_dirs/tf_logs --port=6006```


# Tasks

## Important and Urgent
- [x] implement a validation loop
- [x] add tensorboard 
- [x] extract prediction bin file for local evaluation
- [ ] augment waymo evaluation metrics into training pipeline via python binding to call evaluation APIs

## Important and Not Urgent
- [x] implement resume from
- [x] implement load from
- [x] generate info for mini dataset
- [ ] adopt hydra to manage the messy world of configurations !!!

## Not Important and Urgent

## Not Important and not Urgent
- [x] numba jit warning ! (suppressed)
- [ ] check Kornia, DGL libraries
- [ ] refactor times by implementing a specific class !

## Backlog
- [ ] implement the top view viz from matplotlib
- [ ] add train option for mini dataset
- [ ] remove multi task implememntation
- [ ] change viz to mayavi `REF: https://github.com/DapengFeng/waymo-toolkit`
- [ ] in depth documentation `REF: https://github.com/Jossome/Waymo-open-dataset-document`