import logging

from torch.utils.data import dataloader
from models.readers.voxel_encoder import VoxelFeatureExtractorV3
import random, itertools
import argparse

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    # parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--dry_run", help="execute one pass to make sure everyting is ok !")
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
    data_root = "data/Waymo"
    nsweeps = 1
    train_anno = "data/Waymo/infos_train_01sweeps_filter_zero_gt.pkl"
    val_anno = "data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl"
    test_anno = None

    cfg_total_epochs = 3
    cfg_samples_per_gpu=2
    cfg_workers_per_gpu=2


    optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

    # optimizer
    optimizer = dict(
        type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
    )
    lr_config = dict(
        type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
    )
    working_dir = './work_dirs/dry_run/' #TODO: change to get the cofig file name
    val_interval = 1
    # ============================ End Configurarions section ================================= #

    working_dir = os.path.abspath(working_dir)
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    logger = get_root_logger(working_dir)
    
    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

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

    from torch.utils.data import DataLoader
    from utils.collate import collate_kitti
    
    sampler = None #TODO: check Group Sampler
    batch_size = cfg_samples_per_gpu
    num_workers = cfg_workers_per_gpu
    # data loader 
    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=collate_kitti,
        # pin_memory=True,
        pin_memory=False,
    )

    
    total_steps = cfg_total_epochs * len(data_loader)


    # build detector
    # build backbone, reader, necks and bbox_heads
    model = Config(model)
    
    model = VoxelNet(
        reader = VoxelFeatureExtractorV3(**model.reader),
        backbone = SpMiddleResNetFHD(**model.backbone),
        neck = RPN(**model.neck),
        bbox_head = CenterHead(**model.bbox_head),
        train_cfg=Config(train_cfg),
        test_cfg=Config(test_cfg),
        pretrained=None,
    )
    model.CLASSES = ds.CLASSES

    

    from tools.solver.utils import build_one_cycle_optimizer, _create_learning_rate_scheduler

    cfg_lr_config = Config(lr_config)
    cfg_optimizer = Config(optimizer)
    cfg_optimizer_config = Config(optimizer_config)
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
    
    model.cuda() # train only on GPU
    logger.info(f"model structure: {model}")

    # TODO: handle resume from 
    # TODO: handle load from

    
    trainer = Trainer(model, optimizer, lr_scheduler, working_dir, logger, args, cfg_optimizer_config)
    trainer.run(data_loader=data_loader, max_epochs=cfg_total_epochs)
    


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer():
    def __init__(
        self, 
        model, 
        optimizer,
        lr_scheduler,
        wokring_dir,
        logger,
        run_args,
        optimizer_config
        ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if optimizer_config is not None:
            self.grad_clip = optimizer_config.grad_clip

        # create work_dir
        if not os.path.exists(wokring_dir):
            raise ValueError('Not a valid working dir')
        self.working_dir = wokring_dir
        

        
        self.run_args = run_args
        self.timestamp = get_timestamp()
        self.t = time.time() # timer to measure the iterations and epoch times
        
        if logger is None:
            raise ValueError('Launch the root logger for logging !')
        self.logger = logger
        
        self.log_buffer = LogBuffer()

        self.mode = None # TODO: set property decorator for the following
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

        # hooks definitions
        from trainer_utils import TextLoggerHook
        self.text_logger = TextLoggerHook(interval=10, ignore_last=True, reset_flag=False)

        self.logger.info('Trainer initialized !')



    
    def __call__(self):
        pass

    
    def run(self, data_loader, max_epochs):
        self._max_epochs = max_epochs
        if self.working_dir is None:
            raise ValueError('where is working dir?!')
        self.logger.info('Start running, working dir is {}'.format(self.working_dir))
        self.logger.info('max epochs: {}'.format(self._max_epochs))

        # before_run
        self.text_logger.before_run(self)

        for epoch in range(self._max_epochs):
            self.train(data_loader, epoch)
            self.val()
        # after_run
        


    
    
    def train(self, data_loader, epoch):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self.length = len(data_loader)
        self._max_iters = self._max_epochs * self.length

        base_step = epoch * self.length
        # before train epoch
        self.t = time.time()
        self.text_logger.before_epoch(self)

        for i, batch_data in enumerate(data_loader):
            global_step = base_step + i
            self._inner_iter = i
            if self.lr_scheduler is not None:
                #print(global_step)
                self.lr_scheduler.step(global_step)
            # before_iter , before_train_iter
            self.log_buffer.update({"data_time": time.time() - self.t})

            # batch processor
            # move data to device
            batch_data_tensor = {}
            for k, v in batch_data.items(): 
                if k in [
                    # "anchors", 
                    # "anchors_mask", 
                    # "reg_targets", 
                    # "reg_weights", 
                    # "labels", 
                    "hm", 
                    "anno_box", 
                    "ind", 
                    "mask", 
                    'cat'
                ]:
                    batch_data_tensor[k] = [item.to(device) for item in v] # move dictionaries
                elif k in [ 
                    "voxels",
                    "points",
                    "num_voxels",
                    "num_points",
                    "gt_boxes_and_cls"
                    "coordinates",
                    # "bev_map",
                    # "cyv_voxels",
                    # "cyv_num_voxels",
                    # "cyv_coordinates",
                    # "cyv_num_points",
                ]:
                    batch_data_tensor[k] = v.to(device) # move single items
                else:
                    batch_data_tensor[k] = v # keep the rest on cpu

                # after_data_to_device
                self.log_buffer.update({"transfer_time": time.time() - self.t})

            
            losses = self.model(batch_data_tensor, return_loss=True)
            # after_forward
            self.log_buffer.update({"forward_time": time.time() - self.t})

            log_vars = OrderedDict()
            loss = sum(losses["loss"])
            for loss_name, loss_value in losses.items():
                if loss_name == "loc_loss_elem":
                    log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
                else:
                    log_vars[loss_name] = [i.item() for i in loss_value]
            
            del losses
            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=-1  # TODO: FIX THIS
            )
            # after_parse_loss
            self.log_buffer.update({"loss_parse_time": time.time() - self.t})
            
            if "log_vars" in outputs:
                self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
            self.outputs = outputs
            


            # after_iter , after_train_iter , after_val_iter
            self.optimizer.zero_grad()
            self.outputs["loss"].backward()
            if self.grad_clip is not None:
                clip_grad.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), **self.grad_clip)
            self.optimizer.step()
            self.text_logger.after_train_iter(self)
            self._iter += 1
            self.log_buffer.update({"time": time.time() - self.t})
            self.t = time.time()
            if self.run_args.dry_run == True:
                print('successful train Dry Run !')
                exit()
        
        
        self._epoch += 1 
        # after_train_epoch
        self.text_logger.after_train_epoch(self)
        meta = dict(epoch=self._epoch, iter=self._iter)
        save_checkpoint(self.working_dir, self.model, self.optimizer, meta)
    
    def val(self, data_loader, epoch):
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.logger.info(f"work dir: {self.work_dir}")
        # before_val_epoch
        self.text_logger.before_epoch(self)
        
        detections = {}
        cpu_device = torch.device("cpu")
        for i, batch_data in enumerate(data_loader):
            self._inner_iter = i
            # before_val_iter
            with torch.no_grad():
                # move data to device
                batch_data_tensor = {}
                for k, v in batch_data.item(): 
                    if k in [
                        # "anchors", 
                        # "anchors_mask", 
                        # "reg_targets", 
                        # "reg_weights", 
                        # "labels", 
                        "hm", 
                        "anno_box", 
                        "ind", 
                        "mask", 
                        'cat'
                    ]:
                        batch_data_tensor[k] = [item.to(device) for item in v] # move dictionaries
                    elif k in [ 
                        "voxels",
                        "points",
                        "num_voxels",
                        "num_points",
                        "gt_boxes_and_cls"
                        "coordinates",
                        # "bev_map",
                        # "cyv_voxels",
                        # "cyv_num_voxels",
                        # "cyv_coordinates",
                        # "cyv_num_points",
                    ]:
                        batch_data_tensor[k] = v.to(device) # move single items
                    else:
                        batch_data_tensor[k] = v # keep the rest on cpu
                        
                    outputs = self.model(batch_data_tensor, return_loss=False)
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
                all_predictions = [detections]
                predictions = {}
                for p in all_predictions:
                    predictions.update(p)

                # run the evalutaion function implemented in the dataset class       
                result_dict, _ = self.data_loader.dataset.evaluation(
                    predictions, output_dir=self.work_dir
                ) # TODO: not implemented !!!
                self.logger.info("\n")
                for k, v in result_dict["results"].items():
                    self.logger.info(f"Evaluation {k}: {v}")

                # after_val_epoch
                self.text_logger.after_val_epoch(self)

    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]




def weights_to_cpu(state_dict):
    state_dict_cpu = OrderedDict()
    for k, v in state_dict.items():
        state_dict_cpu[k] = v.cpu()
    return state_dict_cpu



from terminaltables import AsciiTable
def _load_state_dict(module, state_dict):
    """Load state_dict into a module
    """
    unexpected_keys = []
    shape_mismatch_pairs = []

    own_state = module.state_dict()
    for name, param in state_dict.items():
        # a hacky fixed to load a new voxelnet 
        if name not in own_state:
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if param.size() != own_state[name].size():
            shape_mismatch_pairs.append([name, own_state[name].size(), param.size()])
            continue
        own_state[name].copy_(param)

    all_missing_keys = set(own_state.keys()) - set(state_dict.keys())
    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    err_msg = []
    if unexpected_keys:
        err_msg.append(
            "unexpected key in source state_dict: {}\n".format(
                ", ".join(unexpected_keys)
            )
        )
    if missing_keys:
        err_msg.append(
            "missing keys in source state_dict: {}\n".format(", ".join(missing_keys))
        )
    if shape_mismatch_pairs:
        mismatch_info = "these keys have mismatched shape:\n"
        header = ["key", "expected shape", "loaded shape"]
        table_data = [header] + shape_mismatch_pairs
        table = AsciiTable(table_data)
        err_msg.append(mismatch_info + table.table)

    
    if len(err_msg) > 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        
        raise RuntimeError(err_msg)


def load_checkpoint(filename, model):
    if not os.path.isfile(filename):
            raise IOError("{} is not a checkpoint file".format(filename))
    checkpoint = torch.load(filename)
    
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise RuntimeError("No state_dict found in checkpoint file {}".format(filename))

    _load_state_dict(model, state_dict)

def save_checkpoint(working_dir, model, optimizer, meta):
    '''
    meta = dict(epoch=self.epoch + 1, iter=self.iter)
    '''

    if not os.path.exists(working_dir):
        raise NotADirectoryError

    filename = 'epoch_{}.pth'.format(meta['epoch'])
    filepath = os.path.join(working_dir, filename)
    filelink = os.path.join(working_dir, 'latest.pth')
    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict()),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)
    # create symlink to the latest epoch
    if os.path.lexists(filelink):
        os.remove(filelink)
    os.symlink(filepath, filelink)

if __name__ == '__main__':
    
    main()
    