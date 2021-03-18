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

# from models.readers import *
# from models.detectors.voxelnet import VoxelNet
# from models.readers.voxel_encoder import VoxelFeatureExtractorV3
# from models.backbones.scn import SpMiddleResNetFHD
# from models.necks.rpn import RPN
# from models.bbox_heads.center_head import CenterHead

def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)

def build_from_cfg_dict(cfg, registry):
    '''
    example for the configurations that should be found in the yaml file, type is the class type and cfg is the configuration dictionary
    reader:
        type: VoxelFeatureExtractor
        cfg:
            num_input_features: 5
    '''
    # print(OmegaConf.to_yaml(cfg))
    assert isinstance(cfg, DictConfig) 
    assert "type" in cfg
    obj_type = OmegaConf.select(cfg, "type")
    
    # args = Dict(OmegaConf.to_container(cfg.cfg))
    args = cfg.cfg.copy()
    if is_str(obj_type):
        obj_cls = registry.get(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_type))
        )
    if obj_cls is None:
        raise TypeError("cannot identify class type")
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
    elif type == 'val-train': # prepare heatmaps for loss calculations, no augmentation or sampler
        val_train_pipeline = [
            build_from_cfg_dict(stage_configs, PIPELINES) for stage_name, stage_configs in cfg.val_train_pipeline.items()
        ]
        dataset = WaymoDataset(
            info_path=cfg.val_anno,
            root_path=cfg.data_root,
            pipeline=val_train_pipeline,
            # ann_file = cfg.val_anno,
            test_mode=True,
            nsweeps= cfg.nsweeps,
            class_names=cfg.class_names,
        )
    elif type == 'val': # skip heatmaps, not needed in evaluation
        val_pipeline = [
            build_from_cfg_dict(stage_configs, PIPELINES) for stage_name, stage_configs in cfg.val_pipeline.items()
        ]
        dataset = WaymoDataset(
            info_path=cfg.val_anno,
            root_path=cfg.data_root,
            pipeline=val_pipeline,
            # ann_file = cfg.val_anno,
            test_mode=True,
            nsweeps= cfg.nsweeps,
            class_names=cfg.class_names,
        )
    elif type == 'test':
        pass
    else:
        raise ValueError("Unknown data type")

    return dataset

# Note: Does not optimize time if used with PyTorch Dataloader !
import threading
import queue as Queue
# REF: https://github.com/justheuristic/prefetch_generator/blob/master/prefetch_generator/__init__.py
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()
        self.exhausted = False

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        if self.exhausted:
            raise StopIteration
        else:
            next_item = self.queue.get()
            if next_item is None:
                raise StopIteration
            return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

#decorator
class background:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch
    def __call__(self, gen):
        def bg_generator(*args,**kwargs):
            return BackgroundGenerator(gen(*args,**kwargs), max_prefetch=self.max_prefetch)
        return bg_generator


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