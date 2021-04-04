""" GNN Tracking
    Author: Ahmed Bahnasy
"""
import argparse, pickle, os
from viz_predictions import get_obj
from pyquaternion import Quaternion
import numpy as np
from tqdm import tqdm
import copy

from waymo_dataset.waymo_common import _create_pd_detection
from tracking.tracker_gnn import GNNTracker
from tracking.utils import reorganize_info, sort_detections, label_to_name, transform_box


import logging
# A logger for this file
log = logging.getLogger(__name__)

def prepare_predictions(detections, infos):
    """ convert to Waymo coordinates
        append pc in return dictionary
        sort according to time
    """
    ret_list = [] 
    detection_results = {} # copy.deepcopy(detections)

    for token in tqdm(infos.keys()):
        
        detection = detections[token]
        detection_results[token] = copy.deepcopy(detection)
        
        info = infos[token]
        pc_path = info['path']
        ref_obj = get_obj(pc_path)
        pc = ref_obj['lidars']['points_xyz'] # get point cloud
        pc_feat = ref_obj['lidars']['points_feature'] # get point cloud features
        pc = np.concatenate((pc, pc_feat), axis = 1)
        
        anno_path = info['anno_path']
        ref_obj = get_obj(anno_path)
        pose = np.reshape(ref_obj['veh_to_global'], [4, 4]) 

        # from Tensors to Numpy
        # from KITTI to Waymo
        box3d = detection["box3d_lidar"].detach().clone().cpu().numpy() 
        labels = detection["label_preds"].detach().clone().cpu().numpy()
        scores = detection['scores'].detach().clone().cpu().numpy()
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box3d[:, [3, 4]] = box3d[:, [4, 3]]

        box3d_glob = transform_box(box3d, pose)

        frame_id = token.split('_')[3][:-4]
        num_box = len(box3d)
        anno_list =[]
        for i in range(num_box):
            anno = {
                'translation': box3d[i, :3],
                'translation_glob': box3d_glob[i,:3],
                'velocity': box3d[i, [6, 7]],
                'velocity_glob': box3d_glob[i, [6, 7]],
                'detection_name': label_to_name(labels[i]),
                'score': scores[i], 
                'box_id': i,
                'box3d': box3d[i,:],
                'box3d_glob': box3d_glob[i,:]
            }

            anno_list.append(anno)
        
        ret_list.append({
            'token': token, 
            'frame_id':int(frame_id),
            'boxes': anno_list,
            'point_cloud': pc,
            # 'global_boxs': anno_list,
            'timestamp': info['timestamp'] 
        })
    
    sorted_ret_list = sort_detections(ret_list)
    return sorted_ret_list, detection_results

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument("--prediction_results", help="prediction output")
    parser.add_argument("--info_path", type=str)
    parser.add_argument("--checkpoint", help="trained GNN Model")
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--vehicle", type=float, default=5) 
    parser.add_argument("--pedestrian", type=float, default=5)  
    parser.add_argument("--cyclist", type=float, default=5)  
    parser.add_argument("--score_thresh", type=float, default=0.75)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    print('Deploy OK')

    max_dist = {
        'VEHICLE': args.vehicle,
        'PEDESTRIAN': args.pedestrian,
        'CYCLIST': args.cyclist
    }

    # initialize tracker
    tracker = GNNTracker(
        ckpt_path = args.checkpoint,
        max_dist = max_dist, # TODO: refactor later !
        score_thresh = args.score_thresh,
        max_age=args.max_age
    )

    with open(args.prediction_results, 'rb') as f:
        predictions=pickle.load(f)

    with open(args.info_path, 'rb') as f:
        infos=pickle.load(f)
        infos = reorganize_info(infos) #dictionary indexed by token 

    # sort detections, return to Waymo from kitti, append point cloud
    sorted_detections, detection_results = prepare_predictions(predictions, infos)
    
    len_detections = len(sorted_detections)
    
    print("Begin Tracking {} frames\n".format(len_detections))
    

    predictions = {}

    
    
    
    
    # TODO: outer loop for no of epochs
    for i in tqdm(range(len_detections)):
        log.info("===== Processing frame {} in the sequence =====".format(i))
        pred = sorted_detections[i]
        token = pred['token']
        
        
        # reset tracking after one sequence
        if pred['frame_id'] == 0:
            tracker.reset()
            last_time_stamp = pred['timestamp']
        
        time_lag = (pred['timestamp'] - last_time_stamp) 
        last_time_stamp = pred['timestamp']
        
        
        # current detections in global coordinates
        curr_det_boxes = pred['boxes']
        curr_det_pc = pred['point_cloud']
        outputs = tracker.step(curr_det_boxes, curr_det_pc, time_lag)
        tracking_ids = []
        box_ids = [] 

        for item in outputs:
            if item['active'] == 0:
                continue
            
            box_ids.append(item['box_id'])
            tracking_ids.append(item['tracking_id'])
        
        # now reorder 
        detection = detection_results[token]

        remained_box_ids = np.array(box_ids)

        track_result = {} 

        # store box id 
        track_result['tracking_ids']= np.array(tracking_ids)   

        # store box parameter 
        track_result['box3d_lidar'] = detection['box3d_lidar'][remained_box_ids]

        # store box label 
        track_result['label_preds'] = detection['label_preds'][remained_box_ids]

        # store box score 
        track_result['scores'] = detection['scores'][remained_box_ids]

        predictions[token] = track_result
        

    os.makedirs(args.work_dir, exist_ok=True)
    # save prediction files to args.work_dir 
    _create_pd_detection(predictions, infos, args.work_dir, tracking=True)

    result_path = os.path.join(args.work_dir, 'tracking_pred.bin')
    gt_path = os.path.join(args.work_dir, '../gt_preds.bin')

    print("Use Waymo devkit or online server to evaluate the result")
    print("After building the devkit, you can use the following command")
    print("waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main \
           {}  {} ".format(result_path, gt_path))

if __name__ == '__main__':
    main()