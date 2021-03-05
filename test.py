'''
create val dataset
creat model
load checkpoint
model to gpu and eval mode
batch processor
'''
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

import logging
import pickle, copy

import random

from tqdm import tqdm

import numpy as np
import torch


import hydra
from omegaconf import DictConfig, OmegaConf
from addict import Dict

import os, time
from pathlib import Path

from utils.checkpoint import load_checkpoint
from tools.builder import build_dataloader, build_dataset
from models.model_builder import build_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


@hydra.main(config_path="conf", config_name="config")
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

    # build validation dataset
    val_ds = build_dataset(cfg, type='val', logger=logger)
    print(len(val_ds))
    
    # build val loader
    data_loader = build_dataloader(val_ds, 'val', cfg, logger=logger)
    print("datalaoder", len(data_loader.dataset))
    
    
    # build model
    model = build_model(cfg.model, logger=logger)

    if OmegaConf.is_none(cfg, "checkpoint"):
        raise ValueError("Checkpoint must be defined")
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint)
    checkpoint_dir_path = "/".join(checkpoint_path.split("/")[:-1])
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    checkpoint_name = cfg.checkpoint.split('/')[-1].split('.')[0] # ex. epoch_1
    
    
    
    model = model.cuda()
    model.eval()
    mode = "val"

    detections = {}
    cpu_device = torch.device("cpu")
    device = 'cuda' # TODO: move to the top of the file
    tick_inference = time.time()
    for i, batch_data in enumerate(tqdm(data_loader)):
        with torch.no_grad():

            batch_data_tensor = move_batch_to_gpu(batch_data)
            
            outputs = model(batch_data_tensor, return_loss=False)

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
    # output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    output_dir = os.path.join(checkpoint_dir_path, checkpoint_name) # add checkpoint name, create output folder at the same folder of the loaded checkpoint
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_dict, _ = val_ds.evaluation(copy.deepcopy(predictions), output_dir=output_dir)


    


if __name__ =='__main__':
    main()