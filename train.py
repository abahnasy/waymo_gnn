from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

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

from tools.builder import build_dataloader, build_dataset, build_optimizer
from models.model_builder import build_model
from utils.log_buffer import LogBuffer
from utils.checkpoint import load_checkpoint, save_checkpoint
# from utils.config import Config

# import wandb
# wandb.init()

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

# @hydra.main(config_name="conf/config_temp")
@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    
    # get current working directory, use hydra working directory of resume from already existing one
    if not OmegaConf.is_none(cfg, 'resume_from'):
        # get original path
        # original_dir = hydra.utils.get_original_cwd()
        working_dir = "/".join(hydra.utils.to_absolute_path(cfg.resume_from).split('/')[:-1]) # get folder path
        assert os.path.exists(working_dir) == True
    else:
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
    val_ds = build_dataset(cfg, type='val-train', logger=logger)
    print(len(val_ds))
    
    # build val loader
    val_data_loader = build_dataloader(val_ds, 'val', cfg, logger=logger)
    print("datalaoder", len(val_data_loader.dataset))
    
    # build model
    model = build_model(cfg.model, logger=logger)
    print(model)

    # build optimizer
    total_steps = cfg.total_epochs * len(data_loader)
    optimizer, lr_scheduler = build_optimizer(model, cfg.optimizer, cfg.lr, total_steps)

    # move to GPU
    model.cuda() # train only on GPU
    logger.info(f"model structure: {model}")


    
    # TFBoard 
    log_dir = os.path.join(working_dir, 'tf_logs')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    from utils.tf_visualizer import Visualizer as TfVisualizer
    tf_viz_train = TfVisualizer(log_dir, 'train') # logging every epoch
    tf_viz_val = TfVisualizer(log_dir, 'val') # logging every batch
    
    max_epochs = cfg.total_epochs
    batch_val_int = 20 # evaute one batch every 20 training batches
    max_iter = max_epochs*len(data_loader) # TODO: rename to train_data_loader
    _epoch = 0
    _iter = 0
    iter_timer = 0
    
    logger.info('Start running, working dir is {}'.format(working_dir))
    logger.info('max epochs: {}'.format(max_epochs))
    
    # Resume from
    if not OmegaConf.is_none(cfg, 'resume_from'):
        ckpt_path = hydra.utils.to_absolute_path(cfg.resume_from)
        checkpoint = load_checkpoint(model, ckpt_path)
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
        iter_timer = time.time()
        for i, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            torch.cuda.reset_max_memory_allocated()
            logger.info("processing training batch [{}/{}]".format(i, len(data_loader)))
            global_step = base_step + i
            if lr_scheduler is not None:
                lr_scheduler.step(global_step)
            # Move data to GPU
            # batch_data_tensor = {}
            t = time.time()
            batch_data_tensor = move_batch_to_gpu(batch_data)
            print("Timer, moving batch to gpu is {}".format(time.time() -t)); t= time.time()

            # model.forward() does the loss calculations
            time_forward = time.time()
            losses = model(batch_data_tensor, return_loss=True)
            print("elapsed forward + loss time: {}".format(time.time() - time_forward))
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
            # print("elapsed iter time: ", time.time() - iter_timer)
            # Check Dry Run
            if cfg.dry_run:
                print('successful train Dry Run !')
                exit()
            
            # check batch validation
            if (i+1) % batch_val_int == 0:
                model.eval()
                num_eval_batches = 5
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
            log_str = "Epoch [{}/{}], iter: [{}/{}]\tlr: {:.5f}, ".format(
                _epoch,
                max_epochs,
                _iter,
                total_steps,
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


if __name__ == '__main__':
    
    main()
    