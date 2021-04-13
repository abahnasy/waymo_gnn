# Waymo Tracking

#### Requirements

- Linux
- Python >= 3.6
- PyTorch >= 1.4
- CUDA >= 10.0
- CMake >= 3.13.2


## Installation Steps

- Sign up for waymo and accept the terms of use
- install GCloud command line tools and use the same credentials for Waymo dataset
- install Conda
- create conda env from requirement.yml to install dependences
* install MinkowskiEngine
  * follow the steps mentioned in: ``` https://github.com/NVIDIA/MinkowskiEngine```
* install open_waymo_dataset library to extract data
```pip install waymo-open-dataset-tf-2-3-0```
* build cuda layers of dcn and iou3d_nms `bash setup.sh`
* run the downloader to download data splits ``` python3 download_tfrecords.py --split 'training'  --root_path './data/Waymo' ```

## Data Preparations

* preprocess data, extract annotaions and point clouds for every frame into piclke file
  ```
  # train set 
  CUDA_VISIBLE_DEVICES=-1 python3 waymo_dataset/waymo_converter.py --tfrecord_path 'data/Waymo/tfrecord_training/segment-*.tfrecord'  --root_path './data/Waymo/train/'```

  # validation set 
  CUDA_VISIBLE_DEVICES=-1 python3 waymo_dataset/waymo_converter.py --tfrecord_path './data/Waymo/tfrecord_validation/segment-*.tfrecord'  --root_path './data/Waymo/val/'

  # testing set 
  CUDA_VISIBLE_DEVICES=-1 python3 waymo_dataset/waymo_converter.py --tfrecord_path 'data/Waymo/tfrecord_validation/segment-*.tfrecord'  --root_path './data/Waymo/test/'
  ```
* create info files
  ```
  # One Sweep Infos 
  python waymo_dataset/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=1
  python waymo_dataset/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=1
  python waymo_dataset/create_data.py waymo_data_prep --root_path=data/Waymo --split test --nsweeps=1
  # Two Sweep Infos
  python waymo_dataset/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=2
  python waymo_dataset/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=2
  python waymo_dataset/create_data.py waymo_data_prep --root_path=data/Waymo --split test --nsweeps=2
  ```

## Folder Structure
```
.
+-- _config.yml
+-- _drafts
|   +-- begin-with-the-crazy-ideas.textile
|   +-- on-simplicity-in-technology.markdown
+-- _includes
|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html
```

## train cmds
There are two main scripts for training
```python3 ./train.py```

* for training the second stage

```python3 ./train.py model=two_stage```

# test cmds
* adjust ```checkpoint``` in ```./conf/config.yaml``` to refer the trained model checkpoint
* run the prediction ```python3 ./test.py```

# generate ground truth for validation set
```python3 waymo_dataset/waymo_common.py --info_path data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl --result_path data/Waymo/ --gt```



# evaluate predicitons (Detections) !
```python cmd_waymo_eval_kit.py evaluate_detections```

# evaluate predicitons (Tracking) !
```python cmd_waymo_eval_kit.py evaluate_tracking```

# Tracking baseline
```python tracking_baseline.py --work_dir=./output_tracking --checkpoint=./ckpts/epoch_36/prediction.pkl --info_path=./data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl```

# tracking GNN
```python3 tracking_gnn.py --work_dir=./output_tracking --checkpoint=./outputs/2021-04-05/04-37-43/epoch_50.pt --prediction_results=./ckpts/epoch_36/prediction.pkl --info_path=./data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl```

# Launch Tensorboard
```python3 -m tensorboard.main --logdir work_dirs/ --port=6006```


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
- [ ] ~~change viz to mayavi `REF: https://github.com/DapengFeng/waymo-toolkit`~~
- [ ] in depth documentation `REF: https://github.com/Jossome/Waymo-open-dataset-document`
- [ ] open3d viz `https://github.com/caizhongang/waymo_kitti_converter/blob/master/tools/dataloader_visualizer.py`

# Acknowledgment
* `https://github.com/poodarchu/Det3D`
* `https://github.com/open-mmlab/mmdetection`
* `https://github.com/open-mmlab/OpenPCDet`
