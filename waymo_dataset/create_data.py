import copy
from pathlib import Path
import pickle

import fire, os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(MAIN_DIR)


from waymo_dataset.create_gt_database import create_groundtruth_database
from waymo_dataset.waymo_common import create_waymo_infos



def waymo_data_prep(root_path, split, nsweeps=1, sub_sampled=None):
    create_waymo_infos(root_path, split=split, nsweeps=nsweeps, sub_sampled=sub_sampled)
    if split == 'train': 
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )
    

if __name__ == "__main__":
    fire.Fire()