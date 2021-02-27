# Dataset, Dataloader, Model build functions !

import inspect, six

from omegaconf import DictConfig, OmegaConf
from addict import Dict
from torch.utils.data import DataLoader

from tools.solver.utils import build_one_cycle_optimizer, _create_learning_rate_scheduler
from utils.collate import collate_kitti #re-arrange data into tensor
from waymo_dataset.registry import DATASETS, PIPELINES
from waymo_dataset.waymo import WaymoDataset
from waymo_dataset.pipelines import *
# model stages
from models.detectors.voxelnet import VoxelNet
from models.readers.voxel_encoder import VoxelFeatureExtractorV3
from models.backbones.scn import SpMiddleResNetFHD
from models.necks.rpn import RPN
from models.bbox_heads.center_head import CenterHead

def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)

def build_from_cfg_dict(cfg, registry):
    print(OmegaConf.to_yaml(cfg))
    assert isinstance(cfg, DictConfig) and "type" in cfg
    obj_type = cfg.select("type")
    args = Dict(OmegaConf.to_container(cfg.cfg))
    if is_str(obj_type):
        obj_cls = registry.get(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_type))
        )
    return obj_cls(**args)

def build_dataset(cfg, type='train', logger=None):
    if type == 'train':
        train_pipeline = [
            build_from_cfg_dict(stage_configs, PIPELINES) for stage_name, stage_configs in cfg.train_pipeline.items()
        ]
        dataset = WaymoDataset(
            info_path=cfg.train_anno,
            root_path=cfg.data_root,
            pipeline=train_pipeline,
            class_names=cfg.class_names,
            test_mode = False if type == 'train' else True,
            sample = False if type == 'train' else True,
            nsweeps= cfg.nsweeps,
            load_interval = 1
        )
    elif type == 'val':
        train_pipeline = [
            build_from_cfg_dict(stage_configs, PIPELINES) for stage_name, stage_configs in cfg.train_pipeline.items()
        ]
        dataset = WaymoDataset(
            info_path=cfg.val_anno,
            root_path=cfg.data_root,
            pipeline=train_pipeline, #TODO: fix this later, temp use of train pipeline !
            ann_file = cfg.val_anno,
            test_mode=True,
            nsweeps= cfg.nsweeps,
            class_names=cfg.class_names,
        )
    elif type == 'test':
        pass
    else:
        raise ValueError("Unknown data type")

    return dataset

    # ds = WaymoDataset(
    #     info_path = train_anno,
    #     root_path = data_root,
    #     cfg=None,
    #     pipeline=train_pipeline,
    #     class_names=class_names,
    #     test_mode=False,
    #     sample=False,
    #     nsweeps=nsweeps,
    #     load_interval=1,
    # )

def build_dataloader(dataset, type, cfg, logger=None):
    sampler = None # TODO: check Group Sampler
    if type =='train':
        data_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=cfg.num_workers,
            collate_fn=collate_kitti,
            # pin_memory=True,
            pin_memory=False,
        )
    elif type == 'val':
        data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_kitti,
        # pin_memory=True,
        pin_memory=True,
    )
    return data_loader

def build_model(cfg, logger=None):
    reader_cfg = Dict(OmegaConf.to_container(cfg.model.reader))
    backbone_cfg = Dict(OmegaConf.to_container(cfg.model.backbone))
    neck_cfg = Dict(OmegaConf.to_container(cfg.model.neck))
    bbox_head = Dict(OmegaConf.to_container(cfg.model.bbox_head))
    model = VoxelNet(
        reader = VoxelFeatureExtractorV3(**reader_cfg),
        backbone = SpMiddleResNetFHD(**backbone_cfg),
        neck = RPN(**neck_cfg, logger=logger),
        bbox_head = CenterHead(**bbox_head),
        train_cfg= None, # cfg.train_cfg,
        test_cfg=cfg.model.test_cfg,
        pretrained=OmegaConf.is_none(cfg.model, "pretrained")
    )
    # model.CLASSES = ds.CLASSES
    return model

def build_optimizer(model, optim_cfg, lr_config, total_steps):
       

    cfg_lr_config = Dict(OmegaConf.to_container(lr_config))
    cfg_optimizer = Dict(OmegaConf.to_container(optim_cfg))
    # cfg_optimizer_config = Config(optimizer_config)
    if cfg_lr_config.type == "one_cycle":
        # build trainer
        optimizer = build_one_cycle_optimizer(model, cfg_optimizer)
        lr_scheduler = _create_learning_rate_scheduler(
            optimizer, cfg_lr_config, total_steps
        )
        cfg_lr_config = None # TODO: why?!
    else:
        raise NotImplementedError
        optimizer = build_optimizer(model, cfg_optimizer)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.drop_step, gamma=.1)
        # lr_scheduler = None
        cfg_lr_config = None 
    return optimizer, lr_scheduler