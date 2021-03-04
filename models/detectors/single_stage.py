import torch.nn as nn
import logging
from models.registry import DETECTORS
# from .. import builder

from ..frozen_batch_norm import FrozenBatchNorm2d
from utils.checkpoint import load_checkpoint

from models.model_builder import build_backbone, build_bbox_head,build_reader, build_neck


@DETECTORS.register_module
class SingleStageDetector(nn.Module):
    def __init__(
        self,
        reader,
        backbone,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(SingleStageDetector, self).__init__()
        print(reader)
        self.reader = build_reader(reader)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.bbox_head = build_bbox_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            logger = logging.getLogger()
            logger.info("load model from: {}".format(pretrained))
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
            
    def extract_feat(self, data):
        input_features = self.reader(data)
        x = self.backbone(input_features)
        if self.with_neck:
            x = self.neck(x)
        return x


    def forward(self, example, return_loss=True, **kwargs):
        pass

    def predict(self, example, preds_dicts):
        pass

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

    @property
    def with_reader(self):
        # Whether input data need to be processed by Input Feature Extractor
        return hasattr(self, "reader") and self.reader is not None

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, "shared_head") and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, "mask_head") and self.mask_head is not None


    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)