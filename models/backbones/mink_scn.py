"""
"""
""" Minkowski Backbone Trial
"""

import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHD", **kwargs
    ):
        super(SpMiddleResNetFHD, self).__init__()
        # conv input
        self.conv_input = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=num_input_features,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiReLU()
            ) # dimension grid input = output
        # conv1 two residual blocks
        self.conv1 = nn.Sequential( 
            BasicBlock(
                inplanes=16,
                 planes=16,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=3
            ),
             BasicBlock(
                inplanes=16,
                 planes=16,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=3
            )
        )
        # conv2
        self.conv2 = nn.Sequential( 
            ME.MinkowskiConvolution(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),
            BasicBlock(
                inplanes=32,
                 planes=32,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=3
            ),
             BasicBlock(
                inplanes=32,
                 planes=32,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=3
            )
        )
        # conv3
        self.conv3 = nn.Sequential( 
            ME.MinkowskiConvolution(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            BasicBlock(
                inplanes=64,
                 planes=64,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=3
            ),
             BasicBlock(
                inplanes=64,
                 planes=64,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=3
            )
        )
        # conv4
        self.conv4 = nn.Sequential( 
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
            BasicBlock(
                inplanes=128,
                 planes=128,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=3
            ),
             BasicBlock(
                inplanes=128,
                 planes=128,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=3
            )
        )
        # conv extra
        self.conv_extra = nn.Sequential( 

            ME.MinkowskiConvolution(
                in_channels=128,
                out_channels=128,
                kernel_size=(3,1,1),
                stride=(2,1,1),
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU()
            )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.SparseTensor):
        x = self.conv_input(x)
        print("tensor_stride: ", x.tensor_stride)
        x = self.conv1(x)
        print("tensor_stride: ", x.tensor_stride)
        x = self.conv2(x)
        print("tensor_stride: ", x.tensor_stride)
        x = self.conv3(x)
        print("tensor_stride: ", x.tensor_stride)
        x = self.conv4(x)
        print("tensor_stride: ", x.tensor_stride)
        x = self.conv_extra(x)
        print("tensor_stride: ", x.tensor_stride)
        # make dense representation
        dinput, min_coord, tensor_stride = x.dense() #batch+channel+3d
        print("shape", dinput.shape)
        print("min_coord: ", min_coord)
        print("tensor_stride: ", tensor_stride)
        return x



if __name__ == "__main__":
    # from voxels to sparse tensors
    model = SpMiddleResNetFHD(
        num_input_features=128
    )
    model.to(device)

    coords0 = torch.IntTensor(np.random.rand(100000, 3)*10)
    feats0 = torch.FloatTensor(np.random.rand(100000,128))
    
    coords1 = torch.IntTensor(np.random.rand(1000, 3)*10)
    feats1 = torch.FloatTensor(np.random.rand(1000,128))
    

    coords, feats = ME.utils.sparse_collate(
        coords=[coords0, coords1], feats=[feats0, feats1], device=device)

    # sparse tensors
    input = ME.SparseTensor(features=feats, coordinates = coords, device=device)
    # print("tensor_stride: ", input.tensor_stride)
    t = time.time()
    output = model(input)
    print("inference time: ", time.time()- t)
    # print(output.shape)