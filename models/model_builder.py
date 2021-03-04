from tools.builder import build_from_cfg_dict

from models.registry import READERS, DETECTORS, BACKBONES, NECKS, BBOX_HEADS
from models import *

def build_model(cfg, logger=None):
    
    model = build_from_cfg_dict(cfg, DETECTORS)
    # reader_cfg = Dict(OmegaConf.to_container(cfg.model.reader))
    # backbone_cfg = Dict(OmegaConf.to_container(cfg.model.backbone))
    # neck_cfg = Dict(OmegaConf.to_container(cfg.model.neck))
    # bbox_head = Dict(OmegaConf.to_container(cfg.model.bbox_head))
    # model = VoxelNet(
    #     # reader = VoxelFeatureExtractorV3(**reader_cfg),
    #     reader = build_from_cfg_dict(cfg.model.reader, READERS),
    #     backbone = SpMiddleResNetFHD(**backbone_cfg),
    #     neck = RPN(**neck_cfg, logger=logger),
    #     bbox_head = CenterHead(**bbox_head),
    #     train_cfg= None, # cfg.train_cfg,
    #     test_cfg=cfg.model.test_cfg,
    #     pretrained= cfg.model
    # )
    # # model.CLASSES = ds.CLASSES
    return model

def build_reader(cfg):
    return build_from_cfg_dict(cfg, READERS)

def build_backbone(cfg):
    return build_from_cfg_dict(cfg, BACKBONES)

def build_neck(cfg):
    return build_from_cfg_dict(cfg, NECKS)

def build_bbox_head(cfg):
    return build_from_cfg_dict(cfg, BBOX_HEADS)