# REF: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import math
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

# def get_downsample_factor(model_config):
#     try:
#         neck_cfg = model_config["neck"]
#     except:
#         model_config = model_config['first_stage_cfg']
#         neck_cfg = model_config['neck']
#     downsample_factor = np.prod(neck_cfg.get("ds_layer_strides", [1]))
#     if len(neck_cfg.get("us_layer_strides", [])) > 0:
#         downsample_factor /= neck_cfg.get("us_layer_strides", [])[-1]

#     backbone_cfg = model_config['backbone']
#     downsample_factor *= backbone_cfg["ds_factor"]
#     downsample_factor = int(downsample_factor)
#     assert downsample_factor > 0
#     return downsample_factor


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
            "gt_boxes_and_cls",
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

def find_lr(
    model, 
    data_loader, 
    optimizer,
    lr_scheduler=None,
    init_value = 1e-8, 
    final_value=10., 
    beta = 0.98, 
    logger=None, 
    tf_viz_train=None):
    """
    """

    num = len(data_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses_list = []
    log_lrs = []


    max_epochs = 1
    max_iter = max_epochs*len(data_loader) # TODO: rename to train_data_loader
    _epoch = 0
    _iter = 0
    total_steps = 1 * len(data_loader)
    
    
    
    
    model.train()
    # base_step = 0
    # Epoch Training Pass
    # total_log_vars = OrderedDict() # TODO: later implement epoch loss

    for i, batch_data in enumerate(data_loader):
        batch_num += 1
        optimizer.zero_grad()
        torch.cuda.reset_max_memory_allocated()
        logger.info("processing training batch [{}/{}]".format(i, len(data_loader)))
        # global_step = base_step + i
        # if lr_scheduler is not None:
        #     lr_scheduler.step(global_step)
        # Move data to GPU
        # batch_data_tensor = {}
        
        batch_data_tensor = move_batch_to_gpu(batch_data)
        

        # model.forward() does the loss calculations
        
        # with prof.profile("forward pass"):
        losses = model(batch_data_tensor, return_loss=True)
        
        log_vars = OrderedDict()
        loss = sum(losses["loss"])
        
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        tf_viz_train.log_scalar('lr_finder/avg_loss', avg_loss, batch_num) 
        tf_viz_train.log_scalar('lr_finder/smoothed_loss', smoothed_loss, batch_num) 
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses_list
        # Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        tf_viz_train.log_scalar('lr_finder/best_loss', best_loss, batch_num) 
        # Store the values
        losses_list.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        tf_viz_train.log_scalar('lr_finder/log_lrs', log_lrs[-1], batch_num) # get last saved one in the list
        tf_viz_train.log_scalar('lr_finder/lr', lr, batch_num) # actual lr value

        for loss_name, loss_value in losses.items():
            if loss_name == "loc_loss_elem":
                log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
            else:
                log_vars[loss_name] = [i.item() for i in loss_value]
                if len(loss_value) == 1:
                    tf_viz_train.log_scalar('batch_{}'.format(loss_name), loss_value[0].detach().cpu().numpy(), batch_num)
                else:
                    raise ValueError('weird length of the loss !!!')
        del losses
        # outputs = dict(
        #     loss=loss, log_vars=log_vars, num_samples=-1  # TODO: FIX THIS
        # )
        
        # outputs["loss"].backward()
        # with prof.profile("time backward"):
        loss.backward()
        optimizer.step()
        _iter+=1
        

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
        
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses_list



@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    
    # get current working directory, use hydra working directory or resume from already existing one
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
    
    
    # build model
    model = build_model(cfg.model, logger=logger)
    print(model)
    

    # build optimizer
    total_steps = 1 * len(data_loader)
    # optimizer, lr_scheduler = build_optimizer(model, cfg.optimizer, cfg.lr, total_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)
    lr_scheduler = None

    # move to GPU
    model.cuda() # train only on GPU
    logger.info(f"model structure: {model}")


    # TFBoard 
    log_dir = os.path.join(working_dir, 'tf_logs')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    from utils.tf_visualizer import Visualizer as TfVisualizer
    tf_viz_train = TfVisualizer(log_dir, 'train') # logging every epoch


    logs, losses_list = find_lr(model, data_loader, optimizer, logger=logger, tf_viz_train = tf_viz_train)
    import matplotlib.pyplot as plt
    plt.plot(logs[10:-5],losses_list[10:-5])
    plt.savefig("learning_rate_finder.png")


    
       

if __name__ == '__main__':
    
    main()