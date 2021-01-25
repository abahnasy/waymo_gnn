import copy
from pathlib import Path
import pickle

import fire, os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(MAIN_DIR)


from .create_gt_database import create_groundtruth_database
from .waymo_common import create_waymo_infos



def waymo_data_prep(root_path, split, nsweeps=1):
    create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
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