from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

import omegaconf
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import logging
import random, itertools
import argparse
import os, time
from pathlib import Path
from collections import OrderedDict

import hydra
from omegaconf import DictConfig, OmegaConf
from addict import Dict
import numpy as np
import torch

from torch.nn.utils import clip_grad

from tools.builder import build_dataloader, build_dataset, build_model, build_optimizer
from utils.log_buffer import LogBuffer
from utils.checkpoint import load_checkpoint, save_checkpoint
# from utils.config import Config



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _get_max_memory():
        mem = torch.cuda.max_memory_allocated()
        mem_mb = torch.tensor(
            [mem / (1024 * 1024)], dtype=torch.int, device=torch.device("cuda")
        )
        # if trainer.world_size > 1:
        #     dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

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
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    # logger.addHandler(ch)

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


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train a detector")
#     # parser.add_argument("config", help="train config file path")
#     parser.add_argument("--work_dir", help="the dir to save logs and models")
#     parser.add_argument("--resume_from", help="the checkpoint file to resume from")
#     parser.add_argument("--dry_run", help="execute one pass to make sure everyting is ok !")
#     # parser.add_argument("--profiler", help="execute one pass to make sure everyting is ok !")
#     parser.add_argument(
#         "--checkpoint", help="the dir to checkpoint which the model read from"
#     )
#     parser.add_argument(
#         "--validate",
#         action="store_true",
#         help="whether to evaluate the checkpoint during training",
#     )
#     parser.add_argument("--seed", type=int, default=None, help="random seed")
#     args = parser.parse_args()
#     return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def move_batch_to_gpu(batch_data):
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
            if k == 'points': continue #DEBUG: skip loading points to gpu, not needed
            batch_data_tensor[k] = v.to(device) # move single items
        else:
            batch_data_tensor[k] = v # keep the rest on cpu
    return batch_data_tensor

@hydra.main(config_name="configs/config")
def main(cfg : DictConfig) -> None:
    
    # get original path
    original_dir = hydra.utils.get_original_cwd()
    
    # get current working directory
    working_dir = os.getcwd()
    print("Working directory : {}".format(working_dir))
    
    # create logger
    logger = get_root_logger(working_dir)
    
    # set random seeds
    if not OmegaConf.is_none(cfg, "seed"):        
        logger.info("Set random seed to {}".format(cfg.seed))
        set_random_seed(cfg.seed)
    
    # build train dataset
    ds = build_dataset(cfg, type='train', logger=logger)
    print(len(ds))
    
    # build train loader
    data_loader = build_dataloader(ds, 'train', cfg, logger=logger)
    print("dataloader", len(data_loader.dataset))
    
    # build validation dataset
    val_ds = build_dataset(cfg, type='val', logger=logger)
    print(len(val_ds))
    
    # build val loader
    val_data_loader = build_dataloader(val_ds, 'val', cfg, logger=logger)
    print("datalaoder", len(val_data_loader.dataset))
    
    # build model
    model = build_model(cfg, logger=logger)

    # build optimizer
    total_steps = cfg.total_epochs * len(data_loader)
    optimizer, lr_scheduler = build_optimizer(model, cfg.optimizer, cfg.lr, total_steps)

    # move to GPU
    model.cuda() # train only on GPU
    logger.info(f"model structure: {model}")


    
    

    


   


    
    
    
    
    # # ============================ Configurarions section ================================= #
    # # TODO: move to external cfg file
    # tasks = [ # TODO: Remove tasks style
    #     dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST']),
    # ]
    # class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))    
    # # training and testing settings
    # target_assigner = dict(
    #     tasks=tasks,
    # )  
    
    # # model settings
    # model = dict(
    #     type="VoxelNet",
    #     pretrained=None,
    #     reader=dict(
    #         type="VoxelFeatureExtractorV3",
    #         num_input_features=5,
    #     ),
    #     backbone=dict(
    #         type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8),
    #     neck=dict(
    #         type="RPN",
    #         layer_nums=[5, 5],
    #         ds_layer_strides=[1, 2],
    #         ds_num_filters=[128, 256],
    #         us_layer_strides=[1, 2],
    #         us_num_filters=[256, 256],
    #         num_input_features=256,
    #         logger=logging.getLogger("RPN"),
    #     ),
    #     bbox_head=dict(
    #         type="CenterHead",
    #         in_channels=sum([256, 256]),
    #         tasks=tasks,
    #         dataset='waymo',
    #         weight=2,
    #         code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #         common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)}, # (output_channel, num_conv)
    #     ),
    # )

    # # build dataset
    # dataset_type = "WaymoDataset"
    
    # db_sampler = dict(
    #     type="GT-AUG",
    #     enable=False,
    #     db_info_path="data/Waymo/dbinfos_train_1sweeps_withvelo.pkl",
    #     sample_groups=[
    #         dict(VEHICLE=15),
    #         dict(PEDESTRIAN=10),
    #         dict(CYCLIST=10),
    #     ],
    #     db_prep_steps=[
    #         dict(
    #             filter_by_min_num_points=dict(
    #                 VEHICLE=5,
    #                 PEDESTRIAN=5,
    #                 CYCLIST=5,
    #             )
    #         ),
    #         dict(filter_by_difficulty=[-1],),
    #     ],
    #     global_random_rotation_range_per_object=[0, 0],
    #     rate=1.0,
    # ) 
    
    # train_preprocessor = dict(
    #     mode="train",
    #     shuffle_points=True,
    #     global_rot_noise=[-0.78539816, 0.78539816],
    #     global_scale_noise=[0.95, 1.05],
    #     db_sampler=db_sampler,
    #     # db_sampler=None, # TODO: set to None due to error in sample_all()
    #     class_names=class_names,
    # )
    # val_preprocessor = dict(
    #     mode="val",
    #     shuffle_points=False,
    # )
    # voxel_generator = dict(
    #     range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
    #     voxel_size=[0.1, 0.1, 0.15],
    #     max_points_in_voxel=5,
    #     max_voxel_num=150000,
    # )
    # assigner = dict(
    #     target_assigner=target_assigner,
    #     out_size_factor=get_downsample_factor(model),
    #     dense_reg=1,
    #     gaussian_overlap=0.1,
    #     max_objs=500,
    #     min_radius=2,
    # )

    # train_cfg = dict(assigner=assigner)

    # test_cfg = dict(
    #     post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
    #     nms=dict(
    #         use_rotate_nms=True,
    #         use_multi_class_nms=False,
    #         nms_pre_max_size=4096,
    #         nms_post_max_size=500,
    #         nms_iou_threshold=0.7,
    #     ),
    #     score_threshold=0.1,
    #     pc_range=[-75.2, -75.2],
    #     out_size_factor=get_downsample_factor(model),
    #     voxel_size=[0.1, 0.1],
    # )

    # train_pipeline = [
    #     dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    #     dict(type="LoadPointCloudAnnotations", with_bbox=True),
    #     dict(type="Preprocess", cfg=train_preprocessor),
    #     dict(type="Voxelization", cfg=voxel_generator),
    #     dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    #     dict(type="Reformat"),
    # ]

    # train_pipeline = [
    #     LoadPointCloudFromFile(dataset=dataset_type),
    #     LoadPointCloudAnnotations(with_bbox=True),
    #     Preprocess(cfg=Config(train_preprocessor)),
    #     Voxelization(cfg=Config(voxel_generator)),
    #     AssignLabel(cfg=Config(train_cfg["assigner"])),
    #     Reformat()
    # ]

    # test_pipeline = [
    #     LoadPointCloudFromFile(dataset=dataset_type),
    #     LoadPointCloudAnnotations(with_bbox=True),
    #     Preprocess(cfg=Config(val_preprocessor)),
    #     Voxelization(cfg=Config(voxel_generator)),
    #     AssignLabel(cfg=Config(train_cfg["assigner"])),
    #     Reformat()
    # ]

    # data_root = "data/Waymo"
    # nsweeps = 1
    # train_anno = "data/Waymo/infos_train_01sweeps_filter_zero_gt.pkl"
    # val_anno = "data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl"
    # test_anno = None

    # cfg_total_epochs = 36
    # cfg_samples_per_gpu=2
    # cfg_workers_per_gpu=8


    # optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

    # # optimizer
    # optimizer = dict(
    #     type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
    # )
    # lr_config = dict(
    #     type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
    # )
    # working_dir = './work_dirs/dry_run_4/' #TODO: change to get the cofig file name
    # val_interval = 1
    # ============================ End Configurarions section ================================= #

    # working_dir = os.path.abspath(working_dir)
    # Path(working_dir).mkdir(parents=True, exist_ok=True)
    
    
    
    
 

    # if args.work_dir is not None:
    #     pass
    #     # cfg.work_dir = args.work_dir
    

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
    

    # print(len(ds))

    

    # from torch.utils.data import DataLoader
    # from utils.collate import collate_kitti #re-arrange data into tensor
    
    # sampler = None #TODO: check Group Sampler
    # batch_size = cfg_samples_per_gpu
    # num_workers = cfg_workers_per_gpu
    # # data loader 
    # data_loader = DataLoader(
    #     ds,
    #     batch_size=batch_size,
    #     sampler=sampler,
    #     shuffle=(sampler is None),
    #     num_workers=num_workers,
    #     collate_fn=collate_kitti,
    #     # pin_memory=True,
    #     pin_memory=False,
    # )
    # print("dataloader", len(data_loader.dataset))

    
    # total_steps = cfg_total_epochs * len(data_loader)

    # # =========== build val dataset ===========
    # val_ds = WaymoDataset(
    #     root_path=data_root,
    #     info_path=val_anno,
    #     test_mode=True,
    #     ann_file=val_anno,
    #     nsweeps=nsweeps,
    #     class_names=class_names,
    #     pipeline=train_pipeline, #TODO: fix this later, temp use of train pipeline !
    # )
    # print(len(val_ds))
    


    # # data loader 
    # val_data_loader = DataLoader(
    #     val_ds,
    #     batch_size=2,
    #     sampler=sampler,
    #     shuffle=True,
    #     num_workers=2,
    #     collate_fn=collate_kitti,
    #     # pin_memory=True,
    #     pin_memory=True,
    # )
    # print("datalaoder", len(val_data_loader.dataset))
    # # build detector
    # # build backbone, reader, necks and bbox_heads
    # model = Config(model)
    
    # model = VoxelNet(
    #     reader = VoxelFeatureExtractorV3(**model.reader),
    #     backbone = SpMiddleResNetFHD(**model.backbone),
    #     neck = RPN(**model.neck),
    #     bbox_head = CenterHead(**model.bbox_head),
    #     train_cfg=Config(train_cfg),
    #     test_cfg=Config(test_cfg),
    #     pretrained=None,
    # )
    # model.CLASSES = ds.CLASSES

   

    # from tools.solver.utils import build_one_cycle_optimizer, _create_learning_rate_scheduler

    # cfg_lr_config = Config(lr_config)
    # cfg_optimizer = Config(optimizer)
    # cfg_optimizer_config = Config(optimizer_config)
    # if cfg_lr_config.type == "one_cycle":
    #     # build trainer
    #     optimizer = build_one_cycle_optimizer(model, cfg_optimizer)
    #     lr_scheduler = _create_learning_rate_scheduler(
    #         optimizer, cfg_lr_config, total_steps
    #     )
    #     cfg_lr_config = None # TODO: why?!
    # else:
    #     raise NotImplementedError
    #     optimizer = build_optimizer(model, cfg_optimizer)
    #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.drop_step, gamma=.1)
    #     # lr_scheduler = None
    #     cfg_lr_config = None 
    
    # model.cuda() # train only on GPU
    # logger.info(f"model structure: {model}")

    # TODO: handle resume from 
    # TODO: handle load from

    
        
        
    # trainer = Trainer(model, optimizer, lr_scheduler, working_dir, logger, args, cfg_optimizer_config, enable_tf_viz=True)
    # if args.resume_from is not None:
    #     trainer.resume(os.path.join(args.resume_from))
    # trainer.run(train_data_loader=data_loader, val_data_loader= val_data_loader, max_epochs=cfg_total_epochs)
    
    
    # ================= Refactoring !!  ================= 
    # TFBoard 
    log_dir = os.path.join(working_dir, 'tf_logs')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    from utils.tf_visualizer import Visualizer as TfVisualizer
    tf_viz_train = TfVisualizer(log_dir, 'train') # logging every epoch
    tf_viz_val = TfVisualizer(log_dir, 'val') # logging every batch
    
    max_epochs = cfg.total_epochs
    batch_val_int = 100 # evaute one batch every 100 training epochs
    max_iter = max_epochs*len(data_loader) # TODO: rename to train_data_loader
    _epoch = 0
    _iter = 0
    iter_timer = 0
    
    logger.info('Start running, working dir is {}'.format(working_dir))
    logger.info('max epochs: {}'.format(max_epochs))
    
    # Resume from
    if not OmegaConf.is_none(cfg, 'resume_from'):
        checkpoint = load_checkpoint(model, os.path.join(cfg.resume_from))
        _epoch = checkpoint["meta"]["epoch"]
        _iter = checkpoint["meta"]["iter"]
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("resumed epoch %d, iter %d", _epoch, _iter)
    
    # Epochs Loop
    for epoch in range(_epoch, max_epochs):
        logger.info("Staring Epoch {}".format(epoch))
        model.train()
        base_step = epoch * len(data_loader)
        # Epoch Training Pass
        # total_log_vars = OrderedDict() # TODO: later implement epoch loss
        timer_end_iter = time.time()
        for i, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            iter_timer = time.time()
            print("elapsed time for dataloading is {}".format(iter_timer - timer_end_iter))
            torch.cuda.reset_max_memory_allocated()
            logger.info("processing training batch [{}/{}]".format(i, len(data_loader)))
            global_step = base_step + i
            if lr_scheduler is not None:
                lr_scheduler.step(global_step)
            # Move data to GPU
            # batch_data_tensor = {}
            batch_data_tensor = move_batch_to_gpu(batch_data)

            # model.forward() does the loss calculations
            time_forward = time.time()
            losses = model(batch_data_tensor, return_loss=True)
            print("elapsed forward time: {}".format(time.time() - time_forward))
            log_vars = OrderedDict()
            loss = sum(losses["loss"])
            for loss_name, loss_value in losses.items():
                if loss_name == "loc_loss_elem":
                    log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
                else:
                    log_vars[loss_name] = [i.item() for i in loss_value]
                    if len(loss_value) == 1:
                        tf_viz_train.log_scalar('batch_{}'.format(loss_name), loss_value[0].detach().cpu().numpy(), global_step)
                    else:
                        raise ValueError('weird length of the loss !!!')
            del losses
            # outputs = dict(
            #     loss=loss, log_vars=log_vars, num_samples=-1  # TODO: FIX THIS
            # )
            time_backward = time.time()
            # outputs["loss"].backward()
            loss.backward()
            print("elapsed backward time {}".format(time.time() - time_backward))
            if not OmegaConf.is_none(cfg.optimizer, 'grad_clip'):
                clipper_dict = Dict(OmegaConf.to_container(cfg.optimizer.grad_clip))
                clip_grad.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), **clipper_dict)
            optimizer.step()
            _iter+=1
            print("elapsed iter time: ", time.time() - iter_timer)
            # Check Dry Run
            if cfg.dry_run:
                print('successful train Dry Run !')
                exit()
            
            # check batch validation
            if (i+1) % batch_val_int == 0:
                model.eval()
                num_eval_batches = 10
                total_loss = OrderedDict()
                for i, batch_data in enumerate(val_data_loader):
                    if i == num_eval_batches:
                        break
                    logger.info("processing val batch [{}/{}]".format(i, num_eval_batches))
                    with torch.no_grad():
                        # Move data to GPU
                        # batch_data_tensor = {}
                        batch_data_tensor = move_batch_to_gpu(batch_data)
                        # model.forward() does the loss calculations
                        losses = model(batch_data_tensor, return_loss=True)
                        # log_vars = OrderedDict()
                        loss = sum(losses["loss"])
                        for loss_name, loss_value in losses.items():
                            if loss_name == "loc_loss_elem":
                                # log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
                                continue
                            else:
                                assert len([i for i in loss_value]) == 1, 'assertion for tensor board'
                                if not loss_name in total_loss.keys():
                                    total_loss[loss_name] = 0
                                total_loss[loss_name] += loss_value[0].detach().cpu().numpy()
                        del losses
                # log to tensorboad average loss per batch
                for loss_name, loss_value in total_loss.items():
                    tf_viz_val.log_scalar('batch_{}'.format(loss_name), loss_value/num_eval_batches, global_step)
                model.train()
            
            # end iter
            log_str = "Epoch [{}/{}][{}/{}]\tlr: {:.5f}, ".format(
                _epoch,
                max_epochs,
                _iter,
                len(data_loader), #train 
                optimizer.param_groups[0]['lr'],
            )
            log_str += "memory: {}, ".format(_get_max_memory())
            log_str += "loss: {:.5f}".format(log_vars['loss'][0])
            logger.info(log_str)
            timer_end_iter = time.time()
        
        # end epoch
        _epoch += 1
        meta = dict(epoch=_epoch, iter=_iter)
        save_checkpoint(working_dir, model, optimizer, meta)

                        

            
            


        # Epoch Validation Pass

    # ================= End Refactoring !!  ================= 



    






class Trainer():
    def __init__(
        self, 
        model, 
        optimizer,
        lr_scheduler,
        wokring_dir,
        logger,
        run_args,
        optimizer_config,
        enable_tf_viz=False
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

        # TFBoard 
        log_dir = os.path.join(self.working_dir, 'tf_logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        from utils.tf_visualizer import Visualizer as TfVisualizer
        self.enable_tf_viz = enable_tf_viz
        self.tf_viz_train = TfVisualizer(log_dir, 'train') # logging every epoch
        self.tf_viz_val = TfVisualizer(log_dir, 'val') # logging every batch
        # note: append prefix epoch_ or batch_ to distinguish between batch and epoch values
        

        
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
        self._timers = {} # Epoch Timer, Iter Timer, Data Transfer Timer

        # hooks definitions
        from trainer_utils import TextLoggerHook
        self.text_logger = TextLoggerHook(interval=10, ignore_last=True, reset_flag=False)

        self.logger.info('Trainer initialized !')

        self.eval_batch_interval = 10 # Every 10 traing batches, do one validation batch



    
    def __call__(self):
        pass

    def resume(self, checkpoint, resume_optimizer=True, map_location="default"):
        checkpoint = load_checkpoint(self.model, checkpoint)

        self._epoch = checkpoint["meta"]["epoch"]
        self._iter = checkpoint["meta"]["iter"]
        if "optimizer" in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("resumed epoch %d, iter %d", self._epoch, self._iter)

    def run(self, train_data_loader, val_data_loader, max_epochs):
        self._max_epochs = max_epochs
        if self.working_dir is None:
            raise ValueError('where is working dir?!')
        self.logger.info('Start running, working dir is {}'.format(self.working_dir))
        self.logger.info('max epochs: {}'.format(self._max_epochs))

        # before_run
        self.text_logger.before_run(self)

        for epoch in range(self._max_epochs):
            self._timers['train_epoch'] = time.time()
            self.train(train_data_loader, epoch)
            self._timers['train_epoch'] = time.time() - self._timers['train_epoch']
            
            if self.enable_tf_viz:
                self.tf_viz_train.log_scalar('epoch_time', self._timers['train_epoch'], self._epoch) 
            self.val(val_data_loader, epoch)
        # after_run
        


    
    
    def train(self, data_loader, epoch, val_data_loader):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self.length = len(data_loader)
        self._max_iters = self._max_epochs * self.length

        base_step = epoch * self.length
        # before train epoch
        self.t = time.time()
        self.text_logger.before_epoch(self)
        
        total_log_vars = OrderedDict()
        
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
                    if k == 'points': continue #DEBUG: skip loading points to gpu, not needed
                    batch_data_tensor[k] = v.to(device) # move single items
                else:
                    batch_data_tensor[k] = v # keep the rest on cpu

                # after_data_to_device
                self.log_buffer.update({"transfer_time": time.time() - self.t})

            # model.forward() does the loss calculations
            losses = self.model(batch_data_tensor, return_loss=True)
            # after_forward
            self.log_buffer.update({"forward_time": time.time() - self.t})

            log_vars = OrderedDict()
            loss = sum(losses["loss"])
            # self.tf_viz_train.log_scalar('batch_loss', loss.detach().cpu().numpy(), self._iter)
            
            for loss_name, loss_value in losses.items():
                if loss_name == "loc_loss_elem":
                    log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
                else:
                    log_vars[loss_name] = [i.item() for i in loss_value]
                    if self.enable_tf_viz:
                        if len(loss_value) == 1:
                            if not loss_name in total_log_vars.keys():
                                total_log_vars[loss_name] = 0
                            total_log_vars[loss_name] += loss_value[0].detach().cpu().numpy()
                            self.tf_viz_train.log_scalar('batch_{}'.format(loss_name), loss_value[0].detach().cpu().numpy(), self._iter)
                        else:
                            raise ValueError('weird length of the loss !!!')
            
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
        # Log epoch info to tensor board
        if self.enable_tf_viz:
            for k, v in total_log_vars.items():
                self.tf_viz_train.log_scalar('epoch_{}'.format(k), v/len(data_loader.dataset), self._epoch) 
    
    def val_batch(self, data_loader, epoch):
        self.model.eval()
        self.mode = "val"
        
        


    def val(self, data_loader, epoch):
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.logger.info(f"work dir: {self.working_dir}")
        # before_val_epoch
        self.text_logger.before_epoch(self)
        
        # detections = {}
        # cpu_device = torch.device("cpu")
        total_log_vars = OrderedDict()
        for i, batch_data in enumerate(data_loader):
            self._inner_iter = i
            # before_val_iter
            with torch.no_grad():
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

                losses = self.model(batch_data_tensor, return_loss=True)
                log_vars = OrderedDict()
                loss = sum(losses["loss"])
                
                
                for loss_name, loss_value in losses.items():
                    if loss_name == "loc_loss_elem":
                        # total_log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
                        continue
                    else:
                        assert len([i for i in loss_value]) == 1, 'assertion for tensor board'
                        if not loss_name in total_log_vars.keys():
                            total_log_vars[loss_name] = 0
                        total_log_vars[loss_name] += loss_value[0].detach().cpu().numpy()
                        

                      # TODO: mAP evaluation to be done later  
                #     outputs = self.model(batch_data_tensor, return_loss=False)
                #     for output in outputs:
                #         token = output["metadata"]["token"]
                #         for k, v in output.items():
                #             if k not in [
                #                 "metadata",
                #             ]:
                #                 output[k] = v.to(cpu_device)
                #         detections.update(
                #             {token: output,}
                #         )
                # all_predictions = [detections]
                # predictions = {}
                # for p in all_predictions:
                #     predictions.update(p)

                # # run the evalutaion function implemented in the dataset class       
                # result_dict, _ = self.data_loader.dataset.evaluation(
                #     predictions, output_dir=self.work_dir
                # ) # TODO: not implemented !!!
                # self.logger.info("\n")
                # for k, v in result_dict["results"].items():
                #     self.logger.info(f"Evaluation {k}: {v}")

                # after_val_epoch
                self.text_logger.after_val_epoch(self)
                
            
        if self.enable_tf_viz:
            for k, v in total_log_vars.items():
                self.tf_viz_val.log_scalar('epoch_{}'.format(k), v/len(data_loader.dataset), self._epoch)


    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]









if __name__ == '__main__':
    
    main()
    