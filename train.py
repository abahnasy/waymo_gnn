import logging
import random, itertools
import argparse

import numpy as np
import torch

from utils.config import Config
from waymo_dataset.waymo import WaymoDataset
# pipelines
from waymo_dataset.pipelines.loading import LoadPointCloudFromFile, LoadPointCloudAnnotations
from waymo_dataset.pipelines.preprocess import Preprocess
from waymo_dataset.pipelines.voxelization import Voxelization
from waymo_dataset.pipelines.assign_label import AssignLabel
from waymo_dataset.pipelines.formating import Reformat

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    # parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    # cfg = Config.fromfile(args.config)


    if args.work_dir is not None:
        pass
        # cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        pass
        # cfg.resume_from = args.resume_from

    # init logger before other steps
    # logger = get_root_logger(cfg.log_level)
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level="INFO" #level=cfg.log_level
        )
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)
    
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
    data_root = "data/Waymo"
    nsweeps = 1
    train_anno = "data/Waymo/infos_train_01sweeps_filter_zero_gt.pkl"
    val_anno = "data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl"
    test_anno = None
    

    
    ds = WaymoDataset(
        info_path = train_anno,
        root_path = data_root,
        cfg=None,
        pipeline=train_pipeline,
        class_names=class_names,
        test_mode=False,
        sample=False,
        nsweeps=nsweeps,
        load_interval=1,
    )

    print(len(ds))

    # data loader 
    # build detector



if __name__ == '__main__':
    
    main()
    