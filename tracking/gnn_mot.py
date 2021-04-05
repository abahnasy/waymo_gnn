
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import EdgeConv

from tracking.pointnet import PointNetfeat, PointNetCustom
from tracking.lstm import LSTMfeat
from tracking.mlp import TwoLayersMLP, EdgeRegressionMLP
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
    
    def __init__(self, mode = 'train'):
        super(GNNMOT, self).__init__()
        self.appear_extractor = PointNetCustom() # pointnet for appearance extraction
        self.det_motion_extractor = TwoLayersMLP(input_size=9, hidden_size=64, output_size=128) # Two Layer MLP for detected boxes motion feature 
        self.track_motion_extractor = LSTMfeat(input_dim= 9, hidden_dim = 128, n_layers = 2, batch_first=True) # LSTM for for tracked boxes motion features
        self.gnn_conv1 = EdgeConv(256, 256)
        self.gnn_conv2 = EdgeConv(256, 256)
        self.gnn_conv3 = EdgeConv(256, 256)
        self.gnn_conv4 = EdgeConv(256, 256)
        self.edge_regr = EdgeRegressionMLP(input_size=256, hidden_size=64, output_size=1)
        self.mode = mode
        self.triplet_loss_alpha = 10
        self.affinity_ce_loss_criterion = nn.CrossEntropyLoss()
        self.affinity_bce_loss_criterion = nn.BCELoss()

    
    def _compute_layer_loss(self, node_feats, regr_affinity_matrix, gt_affinity_matrix):
        """
        REF: https://arxiv.org/abs/2006.07327 Section 3.3
        """
        log.debug("Entered _compute_layer_loss Function")
        # calculate triplet loss
        N, M = gt_affinity_matrix.shape
        assert node_feats.shape == (N+M, 256)
        # triplet loss
        triplet_loss = torch.zeros(1).cuda()
        for idx in range(M):
            track_obj_feat = node_feats[N+idx, :]
            assert track_obj_feat.shape == torch.Size([256]) # TODO: magic number
            if gt_affinity_matrix[:,idx].sum() == 0: #if match not found
                # log.debug("Not matched track !")
                # minimize over the second term in the triplet loss function only !
                triplet_2 = torch.min(torch.abs(track_obj_feat - node_feats[0:N, :]).sum(dim=1))
                triplet_loss = torch.max(triplet_2 + torch.tensor(self.triplet_loss_alpha), 0)[0] #return value 
                
            else: # if match found
                matched_det_idx = torch.nonzero(gt_affinity_matrix[:,idx], as_tuple=False).item()
                # assert matched_det.shape == 1, "Error in gt affinity matrix, cannot have more than one match" # now handled by .item() function
                matched_det_obj_feat = node_feats[matched_det_idx, :]
                unmatched_det_obj_feat = torch.cat((node_feats[0:matched_det_idx,:],node_feats[matched_det_idx+1:N,:]), dim=0)
                assert unmatched_det_obj_feat.shape[0] == N-1
                unmatched_track_obj_feat = torch.cat((node_feats[N:N+idx,:],node_feats[N+idx+1:N+M,:]), dim=0)
                assert unmatched_track_obj_feat.shape[0] == M-1
                triplet_1 = torch.abs((track_obj_feat - matched_det_idx).sum())
                
                triplet_2 = 0
                if unmatched_det_obj_feat.shape[0] != 0:
                    triplet_2 = torch.min(torch.abs(track_obj_feat - unmatched_det_obj_feat).sum(dim=1))

                triplet_3 = 0
                if unmatched_track_obj_feat.shape[0] != 0:
                    triplet_3 = torch.min(torch.abs(matched_det_obj_feat - unmatched_track_obj_feat).sum(dim=1))
                
                # print(type(matched_det_idx), matched_det_obj_feat.shape)
                # print(unmatched_det_obj_feat.shape)
                triplet_loss += torch.max((triplet_1 - triplet_2 - triplet_3 + torch.tensor(self.triplet_loss_alpha)).clamp(min=0), 0)[0] # return value
        
        # affinity loss
        aff_loss = self.affinity_bce_loss_criterion(regr_affinity_matrix.view(-1,1).unsqueeze(0), gt_affinity_matrix.view(-1,1).unsqueeze(0).float())
        
        # affinity loss calculations 
        # calculations on NxM
        
        not_matched_indices = torch.where(gt_affinity_matrix.sum(dim=1))[0].tolist()
        match_indices = [i for i in range(N) if i not in not_matched_indices]
        if len(match_indices) != 0:
            aff_loss += self.affinity_ce_loss_criterion(regr_affinity_matrix[match_indices, :], gt_affinity_matrix[match_indices,:].argmax(dim=1))
        if len(not_matched_indices) != 0:
            aff_loss += torch.log(torch.exp(regr_affinity_matrix[not_matched_indices,:]).sum())
        # calculations on MxN
        not_matched_indices = torch.where(gt_affinity_matrix.sum(dim=0))[0].tolist()
        match_indices = [i for i in range(M) if i not in not_matched_indices]
        if len(match_indices) != 0:
            aff_loss += self.affinity_ce_loss_criterion(regr_affinity_matrix[:, match_indices].T, gt_affinity_matrix[:, match_indices].argmax(dim=0).T)
        if len(not_matched_indices) != 0:
            aff_loss += torch.log(torch.exp(regr_affinity_matrix[:, not_matched_indices]).sum())
        
        if aff_loss.item() < 0:
            raise ValueError("aff negative loss !!")
        if triplet_loss.item() < 0:
            raise ValueError("triplet negative loss !!")
        return {
            'aff_loss': aff_loss,
            'triplet_loss': triplet_loss,
            'total_loss': aff_loss + triplet_loss
        }
        


    def _compute_affinity_matrix(self, node_feats, N, M):
        # edges = torch.zeros((N*M, 256))
        # c = 0
        # assert node_feats.shape == torch.Size([N+M, 256])
        # for i in range(N):
        #     for j in range(M):
        #         # if affinity_matrix[i, j] != 0:
        #             edges[c] = node_feats[N+j] - node_feats[i] # track - det features
        #             c += 1
        

        # vectorizing h
        edges = (node_feats[N:N+M].reshape(M,256).unsqueeze(0) - node_feats[0:N].reshape(N,256).unsqueeze(1)).reshape(N*M, 256)
        # print("======", edges)
        # print("======", edges_v)
        # print(type(edges), type(edges_v))
        # assert torch.equal(edges.cuda(), edges_v) == True
        
        # exit()
        edges = self.edge_regr(edges)
        return edges.reshape(N,M) # regressed affinity matrix
   
        
    
    def forward(
        self, 
        det_pc_in_box, 
        det_boxes3d, 
        track_pc_in_box, 
        track_boxes3d, 
        init_aff_matrix, 
        gt_affinity_matrix
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
        
        

        
        # === Graph Convolutions === #
        h = self.gnn_conv1(G, graph_feat)
        h = F.relu(h)
        # regr_affinity_matrix = self._compute_affinity_matrix(h, N, M)
        # loss = self._compute_layer_loss(h, regr_affinity_matrix, gt_affinity_matrix)
        
        h = self.gnn_conv2(G, h)
        h = F.relu(h)
        # loss += self._compute_layer_loss(h, regr_affinity_matrix, gt_affinity_matrix)
        
        h = self.gnn_conv3(G, h)
        h = F.relu(h)
        # loss += self._compute_layer_loss(h, regr_affinity_matrix, gt_affinity_matrix)
        
        h = self.gnn_conv4(G, h)
        h = F.relu(h)

        print("Timer: Main forward pass through edge convolution {}".format(time.time() - t)); t = time.time()
        regr_affinity_matrix = self._compute_affinity_matrix(h, N, M)
        print("Timer: regressing affinity matrix {}".format(time.time() - t)); t = time.time()
        # loss += self._compute_layer_loss(h, regr_affinity_matrix, gt_affinity_matrix)
        
        # init_affinity_matrix = graph_adj_matrix[0:N, N:N+M]
        valid_regr_affinity_matrix = torch.mul(init_aff_matrix, regr_affinity_matrix)
        xs,ys= torch.where(valid_regr_affinity_matrix == 0)
        xs,ys=xs.detach(),ys.detach()
        valid_regr_affinity_matrix[xs,ys] = 99 # set to arbitarary large value
        print("Timer: calculating valid affinity matrix {}".format(time.time() - t)); t = time.time()
        
        # eliminate invalid positions from regressed affinity matrix
        assert init_aff_matrix.shape == regr_affinity_matrix.shape
        if self.mode == 'train':
            loss_dict = self._compute_layer_loss(h, regr_affinity_matrix, gt_affinity_matrix)
            return loss_dict['total_loss'], loss_dict['aff_loss'].detach(), loss_dict['triplet_loss'].detach(),valid_regr_affinity_matrix
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





        # TODO: Cosine Similarity, L2, MLP for constructing the affinity matrix
        # implement edge regression and return the matched indices !
        # consturct M x N Affinity Matrix
        # src, dst = G.adj().coalesce().indices()
        # src = src.tolist()
        # dst = dst.tolist()
        # adj_mat = torch.zeros((N+M, N+M)) #TODO: bottle necks !
        # for i in range(len(src)):
        #     if src[i] != dst[i]: #skip self loops
        #         adj_mat[src[i]][dst[i]] = 1
        # # print(adj_mat)
        # # print(adj_mat.sum())
        # affinity_matrix = adj_mat[0:N, N:N+M]
        # print(affinity_matrix.sum())
        # edges = torch.zeros((int(affinity_matrix.sum()), 256))
        # edges = torch.zeros((N*M, 256))
        
        # c = 0
        # for i in range(N):
        #     for j in range(M):
        #         # if affinity_matrix[i, j] != 0:
        #             edges[c] = h[j] - h[i] # track - det features
        #             c += 1

        # afiinity_values = self.edge_regr(edges.cuda())
        # c = 0
        # for i in range(N):
        #     for j in range(M):
        #         # if affinity_matrix[i, j] != 0:
        #             affinity_matrix[i, j] = afiinity_values[c].item()
        #             c += 1
                # else:
                    # affinity_matrix[i, j] = 10 # invalid value
        # print(affinity_matrix.sum()); exit()
