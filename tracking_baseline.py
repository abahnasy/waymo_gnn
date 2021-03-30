import argparse, pickle, os

import numpy as np
from tqdm import tqdm
import copy

from waymo_dataset.waymo_common import _create_pd_detection
from tracking.utils import transform_box, transform_matrix, veh_pos_to_transform, get_obj, reorganize_info, convert_detection_to_global_box
from tracking.tracker import Tracker



def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument("--checkpoint", help="the dir to checkpoint which the model read from")
    parser.add_argument("--info_path", type=str)
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--vehicle", type=float, default=0.8) 
    parser.add_argument("--pedestrian", type=float, default=0.4)  
    parser.add_argument("--cyclist", type=float, default=0.6)  
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

    tracker = Tracker(max_age=args.max_age, max_dist=max_dist, score_thresh=args.score_thresh)

    with open(args.checkpoint, 'rb') as f:
        predictions=pickle.load(f)

    with open(args.info_path, 'rb') as f:
        infos=pickle.load(f)
        infos = reorganize_info(infos)

    global_preds, detection_results = convert_detection_to_global_box(predictions, infos)
    size = len(global_preds)

    print("Begin Tracking {} frames\n".format(size))
    
    predictions = {} 

    for i in tqdm(range(size)):
        pred = global_preds[i]
        token = pred['token']

        # reset tracking after one sequence
        if pred['frame_id'] == 0:
            tracker.reset()
            last_time_stamp = pred['timestamp']

        time_lag = (pred['timestamp'] - last_time_stamp) 
        last_time_stamp = pred['timestamp']

        current_det = pred['global_boxs']

        outputs = tracker.step_centertrack(current_det, time_lag)
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