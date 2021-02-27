'''
create val dataset
creat model
load checkpoint
model to gpu and eval mode
batch processor
'''
import logging
import pickle, copy
from torch.utils.data import dataloader
from models.readers.voxel_encoder import VoxelFeatureExtractorV3
import random, itertools
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.nn.utils import clip_grad

from utils.config import Config
from waymo_dataset.waymo import WaymoDataset
# data pipelines
from waymo_dataset.pipelines.loading import LoadPointCloudFromFile, LoadPointCloudAnnotations
from waymo_dataset.pipelines.preprocess import Preprocess
from waymo_dataset.pipelines.voxelization import Voxelization
from waymo_dataset.pipelines.assign_label import AssignLabel
from waymo_dataset.pipelines.formating import Reformat
# model stages
from models.detectors.voxelnet import VoxelNet
from models.readers.voxel_encoder import VoxelFeatureExtractorV3
from models.backbones.scn import SpMiddleResNetFHD
from models.necks.rpn import RPN
from models.bbox_heads.center_head import CenterHead

import os, time
from pathlib import Path
from collections import OrderedDict

from utils.log_buffer import LogBuffer
from utils.checkpoint import load_checkpoint, save_checkpoint


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def get_root_logger(working_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    filename = "{}.log".format(get_timestamp())
    log_file = os.path.join(working_dir, filename)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def get_downsample_factor(model_config):
    try:
        neck_cfg = model_config["neck"]
    except:
        model_config = model_config['first_stage_cfg']
        neck_cfg = model_config['neck']
    downsample_factor = np.prod(neck_cfg.get("ds_layer_strides", [1]))
    if len(neck_cfg.get("us_layer_strides", [])) > 0:
        downsample_factor /= neck_cfg.get("us_layer_strides", [])[-1]

    backbone_cfg = model_config['backbone']
    downsample_factor *= backbone_cfg["ds_factor"]
    downsample_factor = int(downsample_factor)
    assert downsample_factor > 0
    return downsample_factor


def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)

def parse_args():
    parser = argparse.ArgumentParser(description="Test a detector")
    # parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument("--speed_test", action="store_true")
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true") #TODO: Add support for test set later !
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    # ============================ Configurarions section ================================= #
    # TODO: move to external cfg file
    tasks = [
        dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST']),
    ]
    class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))    
    # training and testing settings
    target_assigner = dict(
        tasks=tasks,
    )  
    
    # model settings
    model = dict(
        type="VoxelNet",
        pretrained=None,
        reader=dict(
            type="VoxelFeatureExtractorV3",
            num_input_features=5,
        ),
        backbone=dict(
            type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8),
        neck=dict(
            type="RPN",
            layer_nums=[5, 5],
            ds_layer_strides=[1, 2],
            ds_num_filters=[128, 256],
            us_layer_strides=[1, 2],
            us_num_filters=[256, 256],
            num_input_features=256,
            logger=logging.getLogger("RPN"),
        ),
        bbox_head=dict(
            type="CenterHead",
            in_channels=sum([256, 256]),
            tasks=tasks,
            dataset='waymo',
            weight=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)}, # (output_channel, num_conv)
        ),
    )

    # build dataset
    dataset_type = "WaymoDataset"
    
    db_sampler = dict(
        type="GT-AUG",
        enable=False,
        db_info_path="data/Waymo/dbinfos_train_1sweeps_withvelo.pkl",
        sample_groups=[
            dict(VEHICLE=15),
            dict(PEDESTRIAN=10),
            dict(CYCLIST=10),
        ],
        db_prep_steps=[
            dict(
                filter_by_min_num_points=dict(
                    VEHICLE=5,
                    PEDESTRIAN=5,
                    CYCLIST=5,
                )
            ),
            dict(filter_by_difficulty=[-1],),
        ],
        global_random_rotation_range_per_object=[0, 0],
        rate=1.0,
    ) 
    
    train_preprocessor = dict(
        mode="train",
        shuffle_points=True,
        global_rot_noise=[-0.78539816, 0.78539816],
        global_scale_noise=[0.95, 1.05],
        # db_sampler=db_sampler,
        db_sampler=None, # TODO: set to None due to error in sample_all()
        class_names=class_names,
    )
    val_preprocessor = dict(
        mode="val",
        shuffle_points=False,
    )
    voxel_generator = dict(
        range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        voxel_size=[0.1, 0.1, 0.15],
        max_points_in_voxel=5,
        max_voxel_num=150000,
    )
    assigner = dict(
        target_assigner=target_assigner,
        out_size_factor=get_downsample_factor(model),
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
    )

    train_cfg = dict(assigner=assigner)

    test_cfg = dict(
        post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
        nms=dict(
            use_rotate_nms=True,
            use_multi_class_nms=False,
            nms_pre_max_size=4096,
            nms_post_max_size=500,
            nms_iou_threshold=0.7,
        ),
        score_threshold=0.1,
        pc_range=[-75.2, -75.2],
        out_size_factor=get_downsample_factor(model),
        voxel_size=[0.1, 0.1],
    )

    # train_pipeline = [
    #     dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    #     dict(type="LoadPointCloudAnnotations", with_bbox=True),
    #     dict(type="Preprocess", cfg=train_preprocessor),
    #     dict(type="Voxelization", cfg=voxel_generator),
    #     dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    #     dict(type="Reformat"),
    # ]

    train_pipeline = [
        LoadPointCloudFromFile(dataset=dataset_type),
        LoadPointCloudAnnotations(with_bbox=True),
        Preprocess(cfg=Config(train_preprocessor)),
        Voxelization(cfg=Config(voxel_generator)),
        AssignLabel(cfg=Config(train_cfg["assigner"])),
        Reformat()
    ]

    test_pipeline = [
        LoadPointCloudFromFile(dataset=dataset_type),
        LoadPointCloudAnnotations(with_bbox=True),
        Preprocess(cfg=Config(val_preprocessor)),
        Voxelization(cfg=Config(voxel_generator)),
        AssignLabel(cfg=Config(train_cfg["assigner"])),
        Reformat()
    ]

    data_root = "data/Waymo"
    nsweeps = 1
    # train_anno = "data/Waymo/infos_train_01sweeps_filter_zero_gt.pkl"
    val_anno = "data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl"
    # test_anno = None

    # cfg_total_epochs = 36
    cfg_samples_per_gpu=1
    cfg_workers_per_gpu=1


    # optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

    # optimizer
    # optimizer = dict(
    #     type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
    # )
    # lr_config = dict(
    #     type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
    # )
    # working_dir = './work_dirs/dry_run/' #TODO: change to get the cofig file name
    val_interval = 1
    # ============================ End Configurarions section ================================= #
    # if args.work_dir is not None:
        # working_dir = args.work_dir

    # working_dir = os.path.abspath(working_dir)
    # Path(working_dir).mkdir(parents=True, exist_ok=True)
    # logger = get_root_logger(working_dir)
    
    
    
    # set random seeds
    # if args.seed is not None:
        # logger.info("Set random seed to {}".format(args.seed))
        # set_random_seed(args.seed)

    
    

    val_ds = WaymoDataset(
        info_path = val_anno,
        root_path = data_root,
        # ann_file=val_anno,
        cfg=None,
        class_names=class_names,
        pipeline=test_pipeline,
        test_mode=True,
        # sample=False,
        nsweeps=nsweeps,
        # load_interval=1,
    )

    print(len(val_ds))

    from torch.utils.data import DataLoader
    from utils.collate import collate_kitti
    
    sampler = None # TODO: check Group Sampler
    batch_size = cfg_samples_per_gpu
    num_workers = cfg_workers_per_gpu
    # data loader 
    data_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=collate_kitti,
        # pin_memory=True,
        pin_memory=False,
    )
    print("dataloader", len(data_loader.dataset))

    
    model = Config(model)
    
    model = VoxelNet(
        reader = VoxelFeatureExtractorV3(**model.reader),
        backbone = SpMiddleResNetFHD(**model.backbone),
        neck = RPN(**model.neck),
        bbox_head = CenterHead(**model.bbox_head),
        train_cfg=None,
        test_cfg=Config(test_cfg),
        pretrained=None,
    )
    model.CLASSES = val_ds.CLASSES

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    checkpoint_name = args.checkpoint.split('/')[-1].split('.')[0] # epoch_1
    
    model = model.cuda()
    model.eval()
    mode = "val"

    detections = {}
    cpu_device = torch.device("cpu")
    device = 'cuda' # TODO: move to the top of the file
    tick_inference = time.time()
    for i, batch_data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            
            # outputs = batch_processor(
            #     model, data_batch, train_mode=False, local_rank=args.local_rank,
            # )
            example_torch = {}
            float_names = ["voxels", "bev_map"]
            for k, v in batch_data.items():
                if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:
                    example_torch[k] = [res.to(device, non_blocking=False) for res in v]
                elif k in [
                    "voxels",
                    "bev_map",
                    "coordinates",
                    "num_points",
                    "points",
                    "num_voxels",
                    #"cyv_voxels",
                    #"cyv_num_voxels",
                    #"cyv_coordinates",
                    #"cyv_num_points"
                ]:
                    example_torch[k] = v.to(device, non_blocking=False)
                # elif k == "calib":
                #     calib = {}
                #     for k1, v1 in v.items():
                #         # calib[k1] = torch.tensor(v1, dtype=dtype, device=device)
                #         calib[k1] = torch.tensor(v1).to(device, non_blocking=False)
                #     example_torch[k] = calib
                else:
                    example_torch[k] = v
            
            outputs = model(example_torch, return_loss=False)

        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )
    inference_time = time.time() - tick_inference
    print("Total inference time is {}, average per frame is {}", inference_time, inference_time/len(val_ds))

    all_predictions = [detections]
    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    # save_pred(predictions, args.work_dir) # TODO: check later
    print("saving prediction bin file")
    output_dir = os.path.join(args.work_dir, checkpoint_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_dict, _ = val_ds.evaluation(copy.deepcopy(predictions), output_dir=output_dir)


    


if __name__ =='__main__':
    main()