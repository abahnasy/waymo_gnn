
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
# from dgl.nn import EdgeConv

from tracking.models.registry import APPEARANCE_FEATURE_EXTRACTORS, MOTION_FEATURE_EXTRACTORS, CONV_OPERATORS
from tools.builder import build_from_cfg_dict
from tracking.models.appearance_feature_extractors.pointnet import PointNetfeat, PointNetCustom
from tracking.models.motion_feature_extractors.lstm import LSTMfeat
from tracking.models.motion_feature_extractors.mlp import TwoLayersMLP
from tracking.models.edge_regressors.edge_regressors_mlp import EdgeRegressionMLP
from utils.bbox import box_np_ops

import logging
# A logger for this file
log = logging.getLogger(__name__)


def verify_matched_indices(matched_indeics):
    """ Test function to make sure a unique id has been assigned
    """
    m_det = {}
    m_track = {}
    for m in matched_indeics:
        if m[0] in m_det.keys():
            return False
        else:
            m_det[m[0]] = True

        if m[1] in m_track.keys():
            return False
        else:
            m_track[m[1]] = True
    return True

class GNNMOT(nn.Module):
    
    def __init__(self, cfg):
        super(GNNMOT, self).__init__()
        self.appear_extractor = PointNetCustom() # pointnet for appearance extraction
        self.det_motion_extractor = TwoLayersMLP(input_size=9, hidden_size=64, output_size=128) # Two Layer MLP for detected boxes motion feature 
        self.track_motion_extractor = LSTMfeat(input_dim= 9, hidden_dim = 128, n_layers = 2, batch_first=True) # LSTM for for tracked boxes motion features
        # self.gnn_conv1 = EdgeConv(256, 256)
        # self.gnn_conv2 = EdgeConv(256, 256)
        # self.gnn_conv3 = EdgeConv(256, 256)
        # self.gnn_conv4 = EdgeConv(256, 256)
        self.edge_regr = EdgeRegressionMLP(input_size=256, hidden_size=64, output_size=1)
        self.graph_conv = build_from_cfg_dict(cfg.graph_conv, CONV_OPERATORS)
        
        self.mode = cfg.mode
        # self.triplet_loss_alpha = 10
        # self.affinity_ce_loss_criterion = nn.CrossEntropyLoss()
        # self.affinity_bce_loss_criterion = nn.BCELoss()
        
    
    def forward(
        self, 
        det_pc_in_box, 
        det_boxes3d, 
        track_pc_in_box, 
        track_boxes3d, 
        init_aff_matrix, 
        gt_affinity_matrix # None in case of eval
    ):
        """
        Args:
            det_list: data of detected objects
            track_list: data of active track lists
            adj: adjacency matrix
        Return:
            matching: [N,2] matched indices
        """
        log.debug("Entered Forward Function")
        t = time.time()

        
        
        
        N = det_pc_in_box.shape[0] #len(det_pc_in_box)
        M = track_pc_in_box.shape[0]
        assert init_aff_matrix.shape == (N,M)
        
        
        det_feats = torch.transpose(det_pc_in_box,2,1)
        print("Timer: calculate det feature {}".format(time.time() - t)); t = time.time()
        
        
        # det_feats = np.zeros((N,5, num_points))
        # for i in range(N):
        #     det_feats[i] = det_pc_in_box[i].T
        # det_feats = torch.from_numpy(det_feats).float()

        track_feats = torch.transpose(track_pc_in_box,2,1)
        print("Timer: calculate track feat {}".format(time.time() - t)); t = time.time()
        # track_feats = np.zeros((M, 5, num_points))
        # for i in range(M):
        #     track_feats[i] = track_pc_in_box[i].T
        # track_feats = torch.from_numpy(track_feats).float()
        
        det_appear_feats = self.appear_extractor(det_feats.float())
        print("Timer: det pointnet {}".format(time.time() - t)); t = time.time()
        det_motion_feats = self.det_motion_extractor(det_boxes3d)
        print("Timer: det motion {}".format(time.time() - t)); t = time.time()
        assert track_feats.shape[0] != 0
        track_appear_feats = self.appear_extractor(track_feats.float())
        print("Timer: track pointnet {}".format(time.time() - t)); t = time.time()
        track_motion_feats = self.track_motion_extractor(track_boxes3d.float())
        print("Timer: track motion {}".format(time.time() - t)); t = time.time()

        det_feats = torch.cat((det_appear_feats, det_motion_feats), dim=1)

        print("Timer: det feature concatenation {}".format(time.time() - t)); t = time.time()
        if self.mode == 'train':
            assert det_feats.requires_grad == True
        track_feats = torch.cat((track_appear_feats, track_motion_feats), dim=1)
        print("Timer: track feature concatenation {}".format(time.time() - t)); t = time.time()
        
        # print(det_feats.shape, track_feats.shape)
        graph_feat = torch.cat((det_feats, track_feats), dim = 0) # appearance and motion concatenated
        print("Timer: Graph feature concatenation {}".format(time.time() - t)); t = time.time()
        if self.mode == 'train':
            assert graph_feat.requires_grad == True
        # print(graph_feat.shape)
        # create graph 
        src = []
        dst = []
        
        # construction of sparse adjacency matrix
        xs, ys = torch.where(init_aff_matrix == 1)
        xs,ys=xs.detach(),ys.detach()
        for i in range(xs.shape[0]):
            src.append(xs[i].item())
            dst.append(N+ys[i].item())
        

        # for i in range(N+M):
        #     for j in range(N+M):
        #         if graph_adj_matrix[i][j] != 0:
        #             src.append(i)
        #             dst.append(j)
        
        print("Timer: graph list formation {}".format(time.time() - t)); t = time.time()
        # log.debug(len(src))
        # log.debug(len(dst))
        # assert len(src) != 0
        # assert len(dst) != 0
        # src = torch.tensor(src, dtype=torch.int64).detach()
        # dst = torch.tensor(dst, dtype=torch.int64).detach()
        

        G = dgl.DGLGraph()
        G.add_nodes(N+M)
        G.add_edges(src, dst)
        G = dgl.add_self_loop(G).to('cuda:0') # consider self features
        print("Timer: Graph constriction {}".format(time.time() - t)); t = time.time()
        
        assert G.num_nodes() == graph_feat.shape[0]
        
        if self.mode =='train':
            losses_dict, regr_affinity_matrix = self.graph_conv(G, graph_feat, gt_affinity_matrix, N, M)
        else:
            regr_affinity_matrix = self.graph_conv(G, graph_feat, gt_affinity_matrix, N, M)
         # init_affinity_matrix = graph_adj_matrix[0:N, N:N+M]
        valid_regr_affinity_matrix = torch.mul(init_aff_matrix, regr_affinity_matrix)
        xs,ys= torch.where(valid_regr_affinity_matrix == 0)
        xs,ys=xs.detach(),ys.detach()
        valid_regr_affinity_matrix[xs,ys] = 99 # set to arbitarary large value
        assert init_aff_matrix.shape == regr_affinity_matrix.shape
        # Train Mode
        if self.mode == 'train':
            return losses_dict['total_loss'], losses_dict['aff_loss'], losses_dict['triplet_loss'], valid_regr_affinity_matrix
        
        # Validation Mode
        else: 
            # matching assingment
            matched_indices = []
            # protected case from parent function !
            # if dist.shape[1] == 0:
            #     return np.array(matched_indices, np.int32).reshape(-1, 2)
            for i in range(N):
                j = valid_regr_affinity_matrix[i].argmin().detach().item()
                if valid_regr_affinity_matrix[i][j] < 99: # any value above sigmoid output to mark as invalid cell
                    valid_regr_affinity_matrix[:, j] = 99 #invalidate this option when choosed
                    matched_indices.append([i, j])
            # return np.array(matched_indices, np.int32).reshape(-1, 2)
            print("Timer: matching {}".format(time.time() - t)); t = time.time()
            assert  verify_matched_indices(matched_indices) == True
            print("Timer: verify matching {}".format(time.time() - t)); t = time.time()
            return np.array(matched_indices, np.int32).reshape(-1, 2)
