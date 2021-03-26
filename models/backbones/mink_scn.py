"""
"""
""" Minkowski Backbone Trial
"""

import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
import time
from models.registry import BACKBONES
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

@BACKBONES.register_module
class MinkowskiSpMiddleResNetFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="MinkowskiSpMiddleResNetFHD", **kwargs
    ):
        super(MinkowskiSpMiddleResNetFHD, self).__init__()
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
                stride=(3,1,1),
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

    def forward(self, features, coordinates, batch_size, input_shape):
        
        # data already received batched
        # coords, feats = ME.utils.sparse_collate(
        # coords=[coords0, coords1], feats=[feats0, feats1], device=device)

        # sparse tensor
        # assert device =="cuda:0"
        x = ME.SparseTensor(features=features, coordinates = coordinates, device=device)
        x = self.conv_input(x)
        # print("="*10)
        # print("tensor shape: {}, tensor_stride: {}".format(x.shape, x.tensor_stride))
        # print(x.coordinates.min(dim=0)[0])
        # print(x.coordinates.max(dim=0)[0])
        x = self.conv1(x)
        # print("="*10)
        # print("tensor shape: {}, tensor_stride: {}".format(x.shape, x.tensor_stride))
        # print(x.coordinates.min(dim=0)[0])
        # print(x.coordinates.max(dim=0)[0])
        x = self.conv2(x)
        # print("="*10)
        # print("tensor shape: {}, tensor_stride: {}".format(x.shape, x.tensor_stride))
        # print(x.coordinates.min(dim=0)[0])
        # print(x.coordinates.max(dim=0)[0])
        x = self.conv3(x)
        # print("="*10)
        # print("tensor shape: {}, tensor_stride: {}".format(x.shape, x.tensor_stride))
        # print(x.coordinates.min(dim=0)[0])
        # print(x.coordinates.max(dim=0)[0])
        x = self.conv4(x)
        # print("="*10)
        # print("tensor shape: {}, tensor_stride: {}".format(x.shape, x.tensor_stride))
        # print(x.coordinates.min(dim=0)[0])
        # print(x.coordinates.max(dim=0)[0])
        x = self.conv_extra(x)
        # print("="*10)
        # print("tensor shape: {}, tensor_stride: {}".format(x.shape, x.tensor_stride))
        # print(x.coordinates.min(dim=0)[0])
        # print(x.coordinates.max(dim=0)[0])
        
        # make dense representation
        min_coordinate = torch.IntTensor((0,0,0))
        output_shape = torch.Size((batch_size,128,2,188,188))
        # output_shape = torch.Size((batch_size,128,2,94,94)) # downsized architecture
        # min_coordinate = min_coordinate.to(device)
        dinput, min_coord, tensor_stride = x.dense(min_coordinate = min_coordinate, shape= output_shape) 
        #batch+channel+3d
        dinput = dinput.view(batch_size, 128*2, 188,188)
        # dinput = dinput.view(batch_size, 128*2, 94, 94)
        # print("shape", dinput.shape)
        # print("min_coord: ", min_coord)
        # print("tensor_stride: ", tensor_stride)
        
        multi_scale_voxel_features = {
            # 'conv1': x_conv1,
            # 'conv2': x_conv2,
            # 'conv3': x_conv3,
            # 'conv4': x_conv4,
        }

        return dinput, multi_scale_voxel_features


# @BACKBONES.register_module
class MinkowskiRCNNSpMiddleFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="MinkowskiRCNNSpMiddleFHD", **kwargs
    ):
        super(MinkowskiRCNNSpMiddleFHD, self).__init__()
        self.name = name
        # self.dcn = None
        self.zero_init_residual = False

        self.middle_conv = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=num_input_features,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution( #downscale
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution( #downscale
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution( #downscale
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(
                in_channels= 64,
                out_channels=64,
                kernel_size=(3,1,1),
                stride=(2,1,1),
                dilation=1,
                bias=False,
                dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU()            
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        # sparse_shape = np.array(input_shape[::-1]) + [0, 0, 1]
        print(voxel_features.shape)
        print(coors.shape)
        input = ME.SparseTensor(features=voxel_features, coordinates = coors, device=device)


        ret = self.middle_conv(input)
        print("ret tensor stride ", ret.tensor_stride)
        ret,_,_ = ret.dense()
        
        
        # min_coordinate = torch.IntTensor((0,0,0))
        # output_shape = torch.Size((2,64,2,188,188))
        # # min_coordinate = min_coordinate.to(device)
        # dinput, min_coord, tensor_stride = ret.dense(min_coordinate = min_coordinate, shape= output_shape) 
        
        print(ret.shape)

        # ret = ret.permute(0, 1, 4, 2, 3).contiguous()
        # N, C, W, D, H = ret.shape
        # ret = ret.view(N, C * W, D, H)

        return ret


if __name__ == "__main__":
    # from voxels to sparse tensors
    model = MinkowskiSpMiddleResNetFHD(
        num_input_features=5,
        ds_factor=8
    )
    # model.to(device)
    sizes = 0
    for p in model.parameters():
        sizes += np.prod(list(p.shape))
    print(sizes)
    # print(backbone)
    model.cuda()
    import os
    cwd = os.getcwd()
    # input_features = torch.load(os.path.join(cwd, 'models/backbones/input_features.pt'))
    # print("features dims", input_features.shape)
    # print(input_features)
    # coors = torch.load(os.path.join(cwd, 'models/backbones/coor.pt'))
    # print("coors dims ", coors.shape)
    # print(coors)
    # batch_mask_0 = np.where(coors[:,0] == 0)
    # batch_mask_1 = np.where(coors[:, 0] == 1)
    # coors = coors[:,1:] # remove batch column
    # coords0 = coors[batch_mask_0].long()
    # print(coords0.shape)
    # feats0 = input_features[batch_mask_0]
    # print(feats0.shape)
    # assert coords0.shape[0] == feats0.shape[0]
    # assert coords0.shape[1] == 3
    # coords1 = coors[batch_mask_1].long()
    # feats1 = input_features[batch_mask_1]
    # assert coords1.shape[0] == feats1.shape[0]
    # assert coords1.shape[1] == 3

    
    # batch_size = torch.load(os.path.join(cwd, 'models/backbones/batch_size.pt'))
    # # print("batch size", batch_size)
    # input_shape = torch.load(os.path.join(cwd, 'models/backbones/input_shape.pt'))
    # print("input_shape", input_shape)
    # exit()
    
    # time_list = []
    # for i in range(1000):
    #     tick = time.time()
    #     model(input_features, coors, batch_size, input_shape)
    #     time_list.append(time.time() - tick)
    
    # print("elapsed time: {}".format(sum(time_list)/len(time_list)))
    # input_features, data["coors"], data["batch_size"], data["input_shape"]

    # coords0 = torch.IntTensor(np.random.rand(100000, 3)*10)
    # feats0 = torch.FloatTensor(np.random.rand(100000,128))
    
    # coords1 = torch.IntTensor(np.random.rand(1000, 3)*10)
    # feats1 = torch.FloatTensor(np.random.rand(1000,128))
    

    # coords, feats = ME.utils.sparse_collate(
    #     coords=[coords0, coords1], feats=[feats0, feats1], device=device)

    # sparse tensors
    # input = ME.SparseTensor(features=feats, coordinates = coords, device=device)
    # print("tensor_stride: ", input.tensor_stride)
    # t = time.time()
    # output = model(input_features.to(device), coors.to(device), batch_size, input_shape)
    # print("inference time: ", time.time()- t)
    # print(output.shape)