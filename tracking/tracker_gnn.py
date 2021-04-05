import logging
import copy, time

import numpy as np
import torch

from tracking.gnn_mot import GNNMOT
from utils.visualizations import get_corners_from_labels_array

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



class GNNTracker():
    
    def __init__(self, ckpt_path, max_dist, score_thresh, max_age) -> None:
        self.model = GNNMOT(mode='eval').cuda()
        self.model.load_state_dict(torch.load(ckpt_path))
        log.info("Trained GNN Model loaded successfully !")
        self.max_age = max_age
        self.WAYMO_TRACKING_NAMES = [
            'VEHICLE',
            'PEDESTRIAN',
            'CYCLIST'
        ]
        self.WAYMO_CLS_VELOCITY_ERROR = max_dist
        self.score_thresh = score_thresh
        self.reset()
    
    
    def reset(self):
        """ used for evaluation
        """
        self.id_count = 0
        self.tracks = [[]]

    
    def step(self, det_boxes, point_cloud, time_lag):
        """ used in inference only 
        Args:
            det_boxes: list of current frame detections
            point_cloud: point cloud of the current frame used to extract appearance features
            time_lag: currently not used # TODO: check for benefit later !
        Returns:

        """
        if len(det_boxes) == 0:
            self.tracks[-1] = []
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
                # det['tracking'] = np.array(det['velocity_glob'][:2]) * -1 *  time_lag # previous positions wrt to time_lag
                det['label_preds'] = self.WAYMO_TRACKING_NAMES.index(det['detection_name'])
                temp.append(det)
        
        processed_det_boxes = temp

        N = len(processed_det_boxes)
        M = len(self.tracks[-1])
        

        # if 'tracking' in processed_det_boxes[0]:
        #     dets = np.array(
        #     [ det['ct'] + det['tracking'].astype(np.float32)
        #     for det in processed_det_boxes], np.float32) # get estimated previous positions [500,2]
        # else:
        dets = np.array( # global positions
            [det['ct'] for det in processed_det_boxes], np.float32) 

        item_cat = np.array([item['label_preds'] for item in processed_det_boxes], np.int32) # N
        track_cat = np.array([track['label_preds'] for track in self.tracks[-1]], np.int32) # M

        max_diff = np.array([self.WAYMO_CLS_VELOCITY_ERROR[box['detection_name']] for box in processed_det_boxes], np.float32)

        tracks = np.array([pre_det['ct'] for pre_det in self.tracks[-1]], np.float32) # M x 2


        if len(tracks) > 0:  # NOT FIRST FRAME
            
            dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
            dist = np.sqrt(dist) # absolute distance in meter

            # invalid links
            init_aff_matrix = ((dist > max_diff.reshape(N, 1)) + (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
            init_aff_matrix =  ~np.array(init_aff_matrix) #valid links
            # print(adj_matrix.sum())
            # adj_matrix = ((dist < max_diff.reshape(N, 1))) > 0 # nearby points according to velocity approximation
            # graph_adj_matrix = np.zeros((N+M, N+M)) # det first then tracks
            # graph_adj_matrix[0:N,N::] = adj_matrix
            # graph_adj_matrix[N::,0:N] = adj_matrix.T
            # print(graph_adj_matrix.sum()); exit()
            det_pc_in_box = [det['pc_in_box'] for det in processed_det_boxes] # N
            det_boxes3d = np.array([box['box3d'] for box in processed_det_boxes])
            det_boxes3d = det_boxes3d.reshape(N, 9)
            track_pc_in_box = [track['pc_in_box'] for track in self.tracks[-1]] #M
            # track_boxes3d = np.array([box['box3d'] for box in self.tracks])
            # track_boxes3d = track_boxes3d.reshape(M, 9) # should be [M,5,9]
            
            
            # ===== LSTM input ===== #
            earliest_frame = int(len(self.tracks)) - 5
            earliest_frame = max(-1, earliest_frame)
            track_boxes3d = torch.zeros(M,5,9) # last five tracks
            
            temp = np.array([track['box3d'] for track in self.tracks[-1]]) # create[N,9] array of boxes
            track_boxes3d[...] = torch.from_numpy(temp).unsqueeze(1) # add latest tracks and repeat them
            counters = [-1]*M # counter for every box
            for t in range(int(len(self.tracks))-1-1,earliest_frame,-1): # skip the latest tracks and loop from the nex to the earliest tracks in T steps
                for idx in range(M):
                    curr_box_id = self.tracks[-1][idx]['tracking_id']
                    if curr_box_id in self.tracks[t][idx]['tracking_id']:
                        # update and repeat, increase counter
                        # try:
                        track_boxes3d[idx, 0:counters[idx],:] = torch.from_numpy(self.tracks[t]['box3d_lidar'][curr_box_id,:])
                        counters[idx] -= 1
            
            assert track_boxes3d.shape[1] == 5 # TODO: magic number !
            assert track_boxes3d.shape[2] == 9
            
            # forward function, pass tensors on GPU
            tick = time.time()
            with torch.no_grad():
                matched_indices = self.model(
                    torch.tensor(det_pc_in_box).cuda(), 
                    torch.from_numpy(det_boxes3d).cuda(), 
                    torch.tensor(track_pc_in_box).cuda(), 
                    track_boxes3d.cuda(), 
                    torch.from_numpy(init_aff_matrix).cuda(), 
                    gt_affinity_matrix = None # not needed in eval mode
                )
            print(" ====== time consumed for inference is {} ====== ".format(time.time() - tick))
            torch.cuda.empty_cache()

        else:  # first few frame
            assert M == 0
            matched_indices = np.array([], np.int32).reshape(-1, 2)

        unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])] #indicies
        

        unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])] #indices
        
        matches = matched_indices

        ret = []
        # for matched detections assign the id of the matched track, add to return
        for m in matches:
            
            track = processed_det_boxes[m[0]] # m[0] det_idx # m[1] track_idx
            track['tracking_id'] = self.tracks[-1][m[1]]['tracking_id']      
            track['age'] = 1
            track['active'] = self.tracks[-1][m[1]]['active'] + 1
            ret.append(track)

        # for unmatched detections, assign a new ID, add to return also.
        for i in unmatched_dets:
            
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
            
            track = self.tracks[-1][i] # work on the latest tracks
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ct = track['ct']

                # movement in the last second
                if 'tracking' in track:
                    offset = track['tracking'] * -1 # move forward #GNN doesn't rely on 'tracking', kept for reference
                    track['ct'] = ct + offset 
                ret.append(track)

        if len(self.tracks) == 1:
            self.tracks[0] = ret # override the empty list in case of first insertion
        else:
            self.tracks.append(ret)
        return ret




            
            
            