import time
import numpy as np
import spconv
from spconv import SparseConv3d, SubMConv3d
from torch import nn
from torch.nn import functional as F

from models.registry import BACKBONES

# from ..build_norm_layer import build_norm_layer

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
}
def build_norm_layer(cfg, num_features, postfix=""):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'SyncBN':
        #     layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )

# REF: https://github.com/traveller59/spconv/issues/30
class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


@BACKBONES.register_module
class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHD", **kwargs
    ):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = spconv.SparseSequential(        
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )


        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):
        """
        Args:
            voxel_features: [num_voxels, 5] from voxel encoder !
            coors: coordinates of the used voxels
            batch_size:
            input_shape: shape of the voxel grid
        Returns:
        """

        # input: # [41, 1600, 1408]
        # Waymo input grid [1504, 1504, 40]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        # sparse_shape = np.array(input_shape[::-1])

        coors = coors.int()
        # t = time.time()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        # print("size after creating sparse Conv", ret.spatial_shape)
        # print("\t\t spconv: creating the sparse vector", time.time() - t); t = time.time()

        x = self.conv_input(ret)
        # print("size after input conv", x.spatial_shape)
        # print("\t\t spconv: input conv", time.time() - t); t = time.time()

        # x_conv1 = self.conv1(x)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)
        x = self.conv1(x)
        # print("x shape after the first conv", x.spatial_shape)
        # print("\t\t spconv: conv 1", time.time() - t); t = time.time()
        x = self.conv2(x)
        # print("size after conv2", x.spatial_shape)
        # print("\t\t spconv: conv 2", time.time() - t); t = time.time()
        x = self.conv3(x)
        # print("size after conv3", x.spatial_shape)
        # print("\t\t spconv: conv 3", time.time() - t); t = time.time()
        x = self.conv4(x)
        # print("size after conv4", x.spatial_shape)
        # print("\t\t spconv: conv 4", time.time() - t); t = time.time()
        

        ret = self.extra_conv(x)
        # print("size after extra_conv", ret.spatial_shape)
        # print(ret.indices.shape)
        # print(ret.indices.min(dim=0))
        # print(ret.indices.max(dim=0))
        # print("\t\t spconv: conv extra", time.time() - t); t = time.time()

        ret = ret.dense()
        # print("\t\t spconv: dense output", time.time() - t); t = time.time()
        # print("outoput from CML after dense is ", ret.shape)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        # print("outoput from CML after mering depth and channel dimensions is ", ret.shape)

        multi_scale_voxel_features = {
            # 'conv1': x_conv1,
            # 'conv2': x_conv2,
            # 'conv3': x_conv3,
            # 'conv4': x_conv4,
        }
        # print("final shape from sparse conv is ", ret.shape); exit()
        return ret, multi_scale_voxel_features


if __name__ == '__main__':
    import torch, time
    
    backbone = SpMiddleResNetFHD(type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8)
    sizes = 0
    for p in backbone.parameters():
        sizes += np.prod(list(p.shape))
    print(sizes)
    # print(backbone)
    backbone.cuda()
