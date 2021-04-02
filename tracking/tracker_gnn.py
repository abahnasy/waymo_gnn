
from dgl.transform import add_edges
from networkx.generators.random_graphs import gnm_random_graph
from networkx.linalg.graphmatrix import adjacency_matrix
from networkx.readwrite.json_graph.adjacency import adjacency_graph
import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import EdgeConv
import copy
import numpy as np
import torch
from tracking.pointnet import PointNetfeat, PointNetCustom
from tracking.lstm import LSTMfeat
from tracking.mlp import TwoLayersMLP, EdgeRegressionMLP
from utils.bbox import box_np_ops
from utils.visualizations import get_corners_from_labels_array
import logging
# A logger for this file
log = logging.getLogger(__name__)

# class LSTMBuffer():
#     """
#     buffer: {
#         track_id: [5,9] 
#     }
#     """
#     def __init__(self, maxsize = 5):
#         #key: tracking_id, value: [[],[],[],[],[]] latest T tracks
#         self.buffer = {} # Every item [N, 9]
#         self.max_size = maxsize
#         self.current_size = 0

#     def push_tracks(self, tracks):
#         """ tracks: [M,9]
#         """
        
#         for track in tracks:
#             tracking_id = track['tracking_id']
#             if tracking_id in self.buffer.keys():
#                 self.buffer[tracking_id] = self.buffer[tracking_id][1::]
#                 self.buffer[tracking_id].append(track['box3d'])
#             else:
#                 # for the first time
#                 self.buffer[tracking_id] = [track['box3d'] for i in range(5)] # repeat five times # TODO: magic number !
    
#     def prepare_gnn_tracking_input(self, tracks):
#         ret = np.zeros((len(tracks), self.max_size, 9)) # TODO: magic number !
#         for i in range(len(tracks)):
#             tracking_id = tracks[i]['tracking_id']
#             ret[i] = self.buffer[tracking_id]
#         return ret




# class TrackerGNN(object):
#     def __init__(self, max_age=0, max_dist={}, score_thresh=0.1):
#         self.max_age = max_age
#         self.WAYMO_CLS_VELOCITY_ERROR = max_dist 
#         self.WAYMO_TRACKING_NAMES = WAYMO_TRACKING_NAMES
#         self.score_thresh = score_thresh 
#         self.gnn_matcher = GNNMOT()
        
#         self.reset()


#     def reset(self):
#         self.id_count = 0
#         self.tracks = []
#         self.lstm_buffer = LSTMBuffer(maxsize=5) # remember the last five tracks

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

def debug(msg): #TODO: should be removed later
    print("DEBUG: {}".format(msg))

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

# ================================ #
# Point Cloud Sampling
# ================================ #
def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

WAYMO_TRACKING_NAMES = [
    'VEHICLE',
    'PEDESTRIAN',
    'CYCLIST'
]
class GNNMOT(nn.Module):
    
    def __init__(self):
        super(GNNMOT, self).__init__()
        self.appear_extractor = PointNetCustom() # pointnet for appearance extraction
        self.det_motion_extractor = TwoLayersMLP(input_size=9, hidden_size=32, output_size=64) # Two Layer MLP for detected boxes motion feature 
        self.track_motion_extractor = LSTMfeat(input_dim= 9, hidden_dim = 64, n_layers = 2, batch_first=True) # LSTM for for tracked boxes motion features
        self.gnn_conv1 = EdgeConv(128, 128)
        self.gnn_conv2 = EdgeConv(128, 128)
        self.gnn_conv3 = EdgeConv(128, 128)
        self.gnn_conv4 = EdgeConv(128, 128)
        self.edge_regr = EdgeRegressionMLP(input_size=128, hidden_size=64, output_size=1)
        self.mode = 'train'
        self.triplet_loss_alpha = 5
        self.affinity_ce_loss_criterion = nn.CrossEntropyLoss()
        self.affinity_bce_loss_criterion = nn.BCELoss()
    
    
    def _compute_layer_loss(self, node_feats, regr_affinity_matrix, gt_affinity_matrix):
        """
        REF: https://arxiv.org/abs/2006.07327 Section 3.3
        """
        log.debug("Entered _compute_layer_loss Function")
        # calculate triplet loss
        N, M = gt_affinity_matrix.shape
        assert node_feats.shape == (N+M, 128)
        # triplet loss
        for idx in range(M):
            track_obj_feat = node_feats[N+idx, :]
            assert track_obj_feat.shape == torch.Size([128]) # TODO: magic number
            if gt_affinity_matrix[:,idx].sum() == 0: #if match not found
                log.debug("Not matched track !")
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
                triplet_loss = torch.max(triplet_1 - triplet_2 - triplet_3 + torch.tensor(self.triplet_loss_alpha), 0)[0] # return value
        
        # affinity loss
        aff_loss = self.affinity_bce_loss_criterion(regr_affinity_matrix.view(-1,1).unsqueeze(0), gt_affinity_matrix.view(-1,1).unsqueeze(0).float())
        
        # affinity loss calculations 
        # calculations on NxM
        try:
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
        except:
            log.debug("NxM calculations")
            log.debug(regr_affinity_matrix[match_indices, :].shape)
            log.debug(gt_affinity_matrix[match_indices,:].shape)
            log.debug("MxN calculations")
            log.debug(regr_affinity_matrix[:, match_indices].shape)
            log.debug(gt_affinity_matrix[:, match_indices].shape)

        return aff_loss + triplet_loss
        


    def _compute_affinity_matrix(self, node_feats, N, M):
        edges = torch.zeros((N*M, 128))
        c = 0
        assert node_feats.shape == torch.Size([N+M, 128])
        for i in range(N):
            for j in range(M):
                # if affinity_matrix[i, j] != 0:
                    edges[c] = node_feats[j] - node_feats[i] # track - det features
                    c += 1
        edges = self.edge_regr(edges)
        return edges.reshape(N,M) # regressed affinity matrix

            
            
            
        
        
        
    
    def forward(
        self, 
        det_pc_in_box, 
        det_boxes3d, 
        track_pc_in_box, 
        track_boxes3d, 
        graph_adj_matrix, 
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

        N = len(det_pc_in_box)
        M = len(track_pc_in_box)
        assert graph_adj_matrix.shape[0] == N+M
        num_points = det_pc_in_box[0].shape[0] if len(det_pc_in_box) != 0 else track_pc_in_box[0].shape[0] # num points in pc subsampled
        
        det_feats = np.zeros((N,5, num_points))
        for i in range(N):
            det_feats[i] = det_pc_in_box[i].T
        det_feats = torch.from_numpy(det_feats).float()

        track_feats = np.zeros((M, 5, num_points))
        for i in range(M):
            track_feats[i] = track_pc_in_box[i].T
        track_feats = torch.from_numpy(track_feats).float()

        det_appear_feats = self.appear_extractor(det_feats)
        det_motion_feats = self.det_motion_extractor(torch.from_numpy(det_boxes3d).float())
        track_appear_feats = self.appear_extractor(track_feats)
        track_motion_feats = self.track_motion_extractor(track_boxes3d.float())

        det_feats = torch.cat((det_appear_feats, det_motion_feats), dim=1)
        track_feats = torch.cat((track_appear_feats, track_motion_feats), dim=1)
        
        # print(det_feats.shape, track_feats.shape)
        graph_feat = torch.cat((det_feats, track_feats), dim = 0) # appearance and motion concatenated
        # print(graph_feat.shape)
        # create graph 
        src = []
        dst = []
        
        for i in range(N+M):
            for j in range(N+M):
                if graph_adj_matrix[i][j] != 0:
                    src.append(i)
                    dst.append(j)
        
        
        # log.debug(len(src))
        # log.debug(len(dst))
        # assert len(src) != 0
        # assert len(dst) != 0
        src = torch.tensor(src)
        dst = torch.tensor(dst)
        
        

        G = dgl.DGLGraph()
        G.add_nodes(N+M)
        G.add_edges(src, dst)
        G = dgl.add_self_loop(G) # consider self features
        
        try:
            assert G.num_nodes() == graph_feat.shape[0]
        except:
            log.debug("number of graph nodes is {}".format(G.num_nodes()))
            log.debug("shape of graph features is {}".format(graph_feat.shape))
            exit()
        

        
        # === Graph Convolutions === #
        h = self.gnn_conv1(G, graph_feat)
        h = F.relu(h)
        regr_affinity_matrix = self._compute_affinity_matrix(h, N, M)
        loss = self._compute_layer_loss(h, regr_affinity_matrix, gt_affinity_matrix)
        h = self.gnn_conv2(G, h)
        h = F.relu(h)
        loss += self._compute_layer_loss(h, regr_affinity_matrix, gt_affinity_matrix)
        h = self.gnn_conv3(G, h)
        h = F.relu(h)
        loss += self._compute_layer_loss(h, regr_affinity_matrix, gt_affinity_matrix)
        h = self.gnn_conv4(G, h)
        h = F.relu(h)
        loss += self._compute_layer_loss(h, regr_affinity_matrix, gt_affinity_matrix)
        # TODO: Cosine Similarity, L2, MLP for constructing the affinity matrix
        # implement edge regression and return the matched indices !
        # consturct M x N Affinity Matrix
        src, dst = G.adj().coalesce().indices()
        src = src.tolist()
        dst = dst.tolist()
        adj_mat = np.zeros((N+M, N+M)) #TODO: bottle necks !
        for i in range(len(src)):
            if src[i] != dst[i]:
                adj_mat[src[i]][dst[i]] = 1
        # print(adj_mat)
        # print(adj_mat.sum())
        affinity_matrix = adj_mat[0:N, N:N+M]
        # print(affinity_matrix.sum())
        # edges = torch.zeros((int(affinity_matrix.sum()), 128))
        edges = torch.zeros((N*M, 128))
        
        c = 0
        for i in range(N):
            for j in range(M):
                # if affinity_matrix[i, j] != 0:
                    edges[c] = h[j] - h[i] # track - det features
                    c += 1

        afiinity_values = self.edge_regr(edges)
        c = 0
        for i in range(N):
            for j in range(M):
                # if affinity_matrix[i, j] != 0:
                    affinity_matrix[i, j] = afiinity_values[c].item()
                    c += 1
                # else:
                    # affinity_matrix[i, j] = 10 # invalid value
        # print(affinity_matrix.sum()); exit()
        if self.mode == 'train':
            return loss, affinity_matrix
        else:
            raise ValueError("Inference mode is not ready yet !!")
            # matching assingment
            matched_indices = []
            # protected case from parent function !
            # if dist.shape[1] == 0:
            #     return np.array(matched_indices, np.int32).reshape(-1, 2)
            for i in range(N):
                print(affinity_matrix[i])
                j = affinity_matrix[i].argmin()
                if affinity_matrix[i][j] < 10: # any value above sigmoid output to mark as invalid cell
                    affinity_matrix[:, j] = 10 #invalidate this option when choosed
                    matched_indices.append([i, j])
            # return np.array(matched_indices, np.int32).reshape(-1, 2)
            print(np.array(matched_indices, np.int32).reshape(-1, 2))
            assert  verify_matched_indices(matched_indices) == True


    def step(self, det_boxes, point_cloud, time_lag):
        """ used in inference only 
        Args:
            det_boxes: list of current frame detections
            point_cloud: point cloud of the current frame used to extract appearance features
        Returns:

        """
        
        if len(det_boxes) == 0:
            self.tracks = []
            return []
        else:
            temp = []
            for det in det_boxes:
             # filter out classes not evaluated for tracking 
                if det['detection_name'] not in self.WAYMO_TRACKING_NAMES:
                    print("filter {}".format(det['detection_name']))
                    continue
                corners = get_corners_from_labels_array(det['box3d'])
                pc_in_box = extract_pc_in_box3d(point_cloud, corners.T)
                if pc_in_box[0].shape[0] == 0:        
                    det['pc_in_box'] = np.zeros((1024,5)) #TODO: magic number !
                else:
                    # subsample the points to a fixed size
                    det['pc_in_box'] = random_sampling(pc_in_box[0], 1024)
                det['ct'] = np.array(det['translation_glob'][:2]) # xy global coordinates
                det['tracking'] = np.array(det['velocity_glob'][:2]) * -1 *  time_lag # previous positions wrt to time_lag
                det['label_preds'] = self.WAYMO_TRACKING_NAMES.index(det['detection_name'])
                temp.append(det)

        processed_det_boxes = temp
        

        N = len(det_boxes)
        M = len(self.tracks)

        # if 'tracking' in processed_det_boxes[0]:
        #     dets = np.array(
        #     [ det['ct'] + det['tracking'].astype(np.float32)
        #     for det in processed_det_boxes], np.float32) # get estimated previous positions [500,2]
        # else:
        dets = np.array( # global positions
            [det['ct'] for det in processed_det_boxes], np.float32) 

        item_cat = np.array([item['label_preds'] for item in processed_det_boxes], np.int32) # N
        track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32) # M

        max_diff = np.array([self.WAYMO_CLS_VELOCITY_ERROR[box['detection_name']] for box in processed_det_boxes], np.float32)

        tracks = np.array([pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2


        if len(tracks) > 0:  # NOT FIRST FRAME
            debug("NOT THE FIRST TIME")
            dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
            dist = np.sqrt(dist) # absolute distance in meter

            # invalid links
            adj_matrix = ((dist > max_diff.reshape(N, 1)) + (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
            adj_matrix =  ~np.array(adj_matrix) #valid links
            # print(adj_matrix.sum())
            # adj_matrix = ((dist < max_diff.reshape(N, 1))) > 0 # nearby points according to velocity approximation
            graph_adj_matrix = np.zeros((N+M, N+M)) # det first then tracks
            graph_adj_matrix[0:N,N::] = adj_matrix
            graph_adj_matrix[N::,0:N] = adj_matrix.T
            # print(graph_adj_matrix.sum()); exit()
            det_pc_in_box = [det['pc_in_box'] for det in processed_det_boxes] # N
            det_boxes3d = np.array([box['box3d'] for box in processed_det_boxes])
            det_boxes3d = det_boxes3d.reshape(N, 9)
            track_pc_in_box = [track['pc_in_box'] for track in self.tracks] #M
            # track_boxes3d = np.array([box['box3d'] for box in self.tracks])
            # track_boxes3d = track_boxes3d.reshape(M, 9) # should be [M,5,9]
            track_boxes3d = self.lstm_buffer.prepare_gnn_tracking_input(self.tracks)
            assert track_boxes3d.shape[1] == 5 # TODO: magic number !
            assert track_boxes3d.shape[2] == 9
            # forward function
            matched_indices = self.gnn_matcher(det_pc_in_box, det_boxes3d, track_pc_in_box, track_boxes3d, graph_adj_matrix)
            raise ValueError("Missing implementation here !!")
        else:  # first few frame
            assert M == 0
            matched_indices = np.array([], np.int32).reshape(-1, 2)

        unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])] #indicies
        debug("Lengh unmatches is {}".format(len(unmatched_dets)))

        unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])] #indices
        
        matches = matched_indices

        ret = []
        # for matched detections assign the id of the matched track, add to return
        for m in matches:
            debug("processing matches")
            track = processed_det_boxes[m[0]] # m[0] det_idx # m[1] track_idx
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']      
            track['age'] = 1
            track['active'] = self.tracks[m[1]]['active'] + 1
            ret.append(track)

        # for unmatched detections, assign a new ID, add to return also.
        for i in unmatched_dets:
            debug("processing unmatched det")
            track = processed_det_boxes[i]
            if track['score'] > self.score_thresh:
                self.id_count += 1
                track['tracking_id'] = self.id_count
                track['age'] = 1
                track['active'] =  1
                ret.append(track)

        # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
        # the object in current frame 
        for i in unmatched_tracks:
            debug("processing unmatched tracks")
            track = self.tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ct = track['ct']

                # movement in the last second
                if 'tracking' in track:
                    offset = track['tracking'] * -1 # move forward #GNN doesn't rely on 'tracking', kept for reference
                    track['ct'] = ct + offset 
                ret.append(track)

        self.tracks = ret
        self.lstm_buffer.push_tracks(ret)
        return ret


            
            
            