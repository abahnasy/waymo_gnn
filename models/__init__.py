from models.backbones.scn import SpMiddleResNetFHD
from models.bbox_heads.center_head import CenterHead
from models.necks.rpn import RPN
from models.readers.voxel_encoder import VoxelFeatureExtractorV3
from models.roi_heads.roi_head import RoIHead
from models.second_stage.bev import BEVFeatureExtractor
# from models.detectors.single_stage import SingleStageDetector
# from models.detectors.two_stage import TwoStageDetector
from models.detectors.voxelnet import VoxelNet