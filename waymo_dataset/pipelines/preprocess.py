import numpy as np
import hydra

from waymo_dataset.registry import PIPELINES
from utils.sampler import preprocess as prep
from utils.bbox import box_np_ops

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, **kwargs):
        self.shuffle_points = kwargs.get("shuffle_points", False) # cfg.shuffle_points
        self.min_points_in_gt = kwargs.get("min_points_in_gt", -1)
        
        self.mode = kwargs.get('mode')
        if self.mode == "train":
            self.global_rotation_noise = kwargs.get("global_rot_noise", None)
            self.global_scaling_noise = kwargs.get("global_scale_noise", None)
            self.class_names = kwargs.get("class_names")
            assert len(self.class_names) != 0
            self.db_sampler = kwargs.get('db_sampler', None)
            if self.db_sampler != None:
                # print(cfg.db_sampler)
                # raise NotImplementedError # TODO: implement the builder !
                # self.db_sampler = build_dbsampler(cfg.db_sampler)
                from utils.sampler.sample_ops import DataBaseSamplerV2
                import pickle, logging
                logger = logging.getLogger("build_dbsampler")
                info_path = hydra.utils.to_absolute_path(self.db_sampler['db_info_path']) # skip hydra current output folder
                with open(info_path, "rb") as f:
                    db_infos = pickle.load(f)
                # build preprocessors
                from utils.sampler.preprocess import DBFilterByDifficulty, DBFilterByMinNumPoint, DataBasePreprocessor
                preprocessors = []    
                if "filter_by_difficulty" in self.db_sampler['db_prep_steps']:
                    v = self.db_sampler['db_prep_steps']["filter_by_difficulty"]
                    preprocessors.append(DBFilterByDifficulty(v, logger=logger))
                elif "filter_by_min_num_points" in self.db_sampler['db_prep_steps']:
                    v = self.db_sampler['db_prep_steps']["filter_by_min_num_points"]
                    preprocessors.append(DBFilterByMinNumPoint(v, logger=logger))
                db_prepor = DataBasePreprocessor(preprocessors)
                self.db_sampler = DataBaseSamplerV2(
                    db_infos, 
                    groups = self.db_sampler['sample_groups'],
                    db_prepor = db_prepor, 
                    rate = self.db_sampler['rate'], 
                    global_rot_range = self.db_sampler['global_random_rotation_range_per_object'], 
                    logger=logger
                )
            else:
                self.db_sampler = None 
                
            self.npoints = kwargs.get("npoints", -1)

        self.no_augmentation = kwargs.get('no_augmentation', False)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        else:
            raise NotImplementedError

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }

        if self.mode == "train" and not self.no_augmentation:
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= self.min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )


                    points = np.concatenate([sampled_points, points], axis=0)

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
            
            gt_dict["gt_boxes"], points = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            )
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )
        elif self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes


        if self.shuffle_points:
            np.random.shuffle(points)

        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info