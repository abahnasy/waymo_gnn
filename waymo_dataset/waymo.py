""" Adapted from https://github.com/tianweiy/CenterPoint
"""
from pathlib import Path
import pickle
from torch.utils.data import Dataset
from waymo_dataset.pipelines.compose import Compose

class WaymoDataset(Dataset):
    
    CLASSES = None
    NumPointFeatures = 5  # x, y, z, intensity, elongation

    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        sample=False,
        nsweeps=1,
        load_interval=1,
        **kwargs,
    ):
        
        super(WaymoDataset, self).__init__(
            
        )
        self._info_path = info_path
        self._root_path = Path(root_path)
        self._class_names = class_names
        self.test_mode = test_mode

        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)
            
        self.load_interval = load_interval 
        self.sample = sample
        self.nsweeps = nsweeps
        print("Using {} sweeps".format(nsweeps))

        self._info_path = info_path
        self._class_names = class_names
        self._num_point_features = WaymoDataset.NumPointFeatures if nsweeps == 1 else WaymoDataset.NumPointFeatures+1

    def reset(self):
        assert False 

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _waymo_infos_all = pickle.load(f)

        self._waymo_infos = _waymo_infos_all[::self.load_interval]

        print("Using {} Frames".format(len(self._waymo_infos)))

    def __len__(self):

        if not hasattr(self, "_waymo_infos"):
            self.load_infos(self._info_path)

        return len(self._waymo_infos)

    def get_sensor_data(self, idx):
        info = self._waymo_infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps, 
            },
            "metadata": {
                "image_prefix": self._root_path, # TODO: refactor, used to get root path in dbsampler
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None, #TODO: remove param
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "WaymoDataset",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None):
        from .waymo_common import _create_pd_detection, reorganize_info

        infos = self._waymo_infos 
        infos = reorganize_info(infos)

        _create_pd_detection(detections, infos, output_dir)

        print("use waymo devkit tool for evaluation")

        return None, None 