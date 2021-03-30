# Author: Ahmed Bahnasy
"""Visualize predictions vs gt
Args:
    path to predictions.pkl
    path to gt (if needed)
"""
import argparse
from multiprocessing import Value
import pickle
import os

import numpy as np
from utils.visualizations import write_points_ply_file, write_oriented_bbox, create_bev_view
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
TYPE_LIST = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

def get_obj(path):
    path = os.path.join(MAIN_DIR, path)
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction Visualizations")
    parser.add_argument("--prediction_pkl", help="Predictions.pkl file", default ="./ckpts/prediction.pkl")
    parser.add_argument("--val_info_file", help="validation info file for loading gt", default = "./data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl")
    parser.add_argument("--classes", help="classes to consider", default=['VEHICLE', 'PEDESTRIAN', 'CYCLIST'])
    parser.add_argument("--score_threshold", help="threshold for prediction scores", default=0.60)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.prediction_pkl, 'rb') as f:
        predictions = pickle.load(f) # dictionary with seq_x_frame_x.pkl as keys
    len_predictions = len(predictions)
    with open(args.val_info_file, 'rb') as f:
        gt = pickle.load(f) # list of dictionaries, every dictionary has token = seq_x_frame_x.pkl
    len_gt = len(gt)
    assert len_gt == len_predictions
    
    
    # loop over every dictionary in the gt and get the respective predictions using token key
    for gt_frame in gt:
        token = gt_frame['token']
        gt_file_path = gt_frame['path']
        gt_info = get_obj(gt_file_path)
        points = gt_info['lidars']['points_xyz'] # get xyz points
        abs_ply_path = os.path.join(MAIN_DIR, 'viz', 'pc_{}.ply'.format(token.split('.')[0])) # remove .pkl extension
        # write_points_ply_file(points,abs_ply_path)
        # get gt_bboxes
        gt_boxes = gt_frame['gt_boxes']
        # return from KITTI to Waymo
        gt_boxes[:, -1] = -gt_boxes[:, -1] - np.pi / 2
        gt_boxes[:,[3,4]] = gt_boxes[:, [4, 3]]

        gt_names = gt_frame['gt_names']
        # filter unwanted classes
        mask = [True if gt_name in args.classes else False for gt_name in gt_names]
        gt_boxes = gt_boxes[mask]
        gt_names = gt_names[mask]
        ply_abs_path = os.path.join(MAIN_DIR, 'viz', 'gt_bboxes_{}.ply'.format(token.split('.')[0]))
        # write_oriented_bbox(gt_boxes, ply_abs_path)
        #predictions
        pred_frame = predictions[token]
        pred_scores = pred_frame['scores']
        indices = np.where(pred_scores >= args.score_threshold)
        pred_boxes = pred_frame['box3d_lidar'][indices].numpy()
        #from KITTI to Waymo
        pred_boxes[:, -1] = -pred_boxes[:, -1] - np.pi / 2
        pred_boxes[:,[3,4]] = pred_boxes[:, [4, 3]]

        pred_names = pred_frame['label_preds'][indices].numpy()
        pred_names = [TYPE_LIST[i] for i in pred_names] # from num to laebl str
        ply_abs_path = os.path.join(MAIN_DIR, 'viz', 'pred_bboxes_{}.ply'.format(token.split('.')[0]))
        # write_oriented_bbox(pred_boxes, ply_abs_path)

        # ===== BEV View ===== #
        bev_img_abs_path = os.path.join(MAIN_DIR, 'viz', 'bev_{}.png'.format(token.split('.')[0]))
        create_bev_view(points, gt_boxes, gt_names, pred_boxes, pred_names, bev_img_abs_path)

        break
            

if __name__ == "__main__":
    main() # TODO: 3D visualizations of bboxes using open3d

"""
predictions.pkl
    type: dict
    keys: tokens
    values: dict_keys(['box3d_lidar', 'scores', 'label_preds', 'metadata'])

val_info_file.pkl
    type: list
    list_type: dict, values:dict_keys(['path', 'anno_path', 'token', 'timestamp', 'sweeps', 'gt_boxes', 'gt_names'])
"""


