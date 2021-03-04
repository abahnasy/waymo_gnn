from tools.registry import Registry

READERS = Registry("readers")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
BBOX_HEADS = Registry("bbox_heads")
DETECTORS = Registry("detectors")
SECOND_STAGE = Registry("second_stage")
ROI_HEAD = Registry("roi_head")