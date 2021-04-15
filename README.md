# Waymo Detection and Tracking

## Requirements

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
* install Waymo dataset evaluation kit. Follow the steps mentioned in https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md

- optional: sparse convolutions are implemented using MinkowskiEngine and spconv library. we adopted MinkowskiEngine due to its efficiency and spped. in case you are interested in running spconv. you need to follow these steps
  ```
  sudo apt-get install libboost-all-dev
  git clone https://github.com/traveller59/spconv.git --recursive
  cd spconv && git checkout 7342772
  python setup.py bdist_wheel
  cd ./dist && pip install *
  ```
  - then apply the following fixes:
    - fix build error in spconv https://github.com/pytorch/extension-script/issues/6
    - solve cuda compiler error https://github.com/traveller59/spconv/issues/211
    - fix cudnn version check while building spconv https://github.com/pytorch/pytorch/issues/40965


## Data Preparation

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
├── cmd_waymo_eval_kit.py     # script to run Waymo evaluation kit !
├── conf                      # configurations for the detector
├── conf_tracking             # configurations for the tracker
├── data
├── download_tfrecords.py
├── outputs
├── runs
├── setup.sh
├── test.py
├── tracking
├── tracking_baseline.py       # inference script for the baseline tracker
├── tracking_gnn.py            # inference script for the GNN Tracker
├── train.py                   # training script for the detector
├── train_tracker.py           # training script for the tracker
├── trainer_utils.py
└── viz_predictions.py
```

## Training and Evaluation commands
### Detector
All configurations for the Detector is saved in `./conf` folder inside `*.yml` files.
#### Train
- ```python3 ./train.py```
#### Evaluate
* adjust ```checkpoint``` in ```./conf/config.yaml``` to refer the trained model checkpoint
* run the prediction
  ```
  python ./test.py
  ```
* generate Ground Truth data for the validation set
  ```
  python waymo_dataset/waymo_common.py --info_path data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl --result_path data/Waymo/ --gt
  ```
* evaluate predicitons (Detections) !
  ```
  python cmd_waymo_eval_kit.py evaluate_detections
  ```
  before running the evaluation cmd, open `cmd_waymo_eval_kit.py` and insert the correct links the prediction file and ground truth file generated from the model and the current used validation dataset.

### Tracker
#### Train
All configurations for the Detector is saved in `./conf_tracking` folder inside `*.yml` files.
- ```python ./train_tracker.py ```
#### Evaluate
* Tracking baseline: this is the baseline tracker which uses velocity cues and greedy matching to assign new detections to the current tracks
  ```
  python tracking_baseline.py --work_dir=./output_tracking --checkpoint=./ckpts/epoch_36/prediction.pkl --info_path=./data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl
  ```
  provide the arguments for the current checkpoint resulted from training and dataset info file generated when preparing the data.

* Tracking GNN
  * adjust ```checkpoint``` in ```./conf/config.yaml``` to refer the trained model checkpoint
  ```
  python3 tracking_gnn.py --work_dir=./output_tracking --checkpoint=./outputs/2021-04-05/04-37-43/epoch_50.pt --prediction_results=./ckpts/epoch_36/prediction.pkl --info_path=./data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl
  ```
  provide the arguments for the current checkpoint resulted from training and dataset info file generated when preparing the data.


* evaluate predicitons (Tracking) !
  ```
  python cmd_waymo_eval_kit.py evaluate_tracking 
  ```
  before running the evaluation cmd, open `cmd_waymo_eval_kit.py` and insert the correct links the prediction file and ground truth file generated from the model and the current used validation dataset.


# Launch Tensorboard
```python3 -m tensorboard.main --logdir work_dirs/ --port=6006```


# Acknowledgment
* https://github.com/poodarchu/Det3D
* https://github.com/open-mmlab/mmdetection
* https://github.com/open-mmlab/OpenPCDet
* https://github.com/tianweiy/CenterPoint