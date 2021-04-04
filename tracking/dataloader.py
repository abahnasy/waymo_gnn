import logging, pickle
import torch
from torch._C import Value
from torch.utils.data.dataset import Dataset
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from tracking.utils import reorganize_info, transform_box
from viz_predictions import get_obj
from utils.visualizations import get_corners_from_labels_array

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

# A logger for this file
log = logging.getLogger(__name__)
class TrackerDataset(Dataset):
    """
    """

    def __init__(self, info_path):
        super(TrackerDataset, self).__init__()

        self.WHITELIST = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST'] #TODO: move to config #equivalent to WAYMO_TRACKING_NAMES
        self.TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
        self.max_dist = {
            'VEHICLE': 5,
            'PEDESTRIAN': 1,
            'CYCLIST': 1
        }

        with open(hydra.utils.to_absolute_path(info_path), 'rb') as f:
            infos = pickle.load(f)
            infos = reorganize_info(infos) # dictionary indexed by tokens

        self.train_data_extract = [] # data extracted and prepared for training
        log.info("Extracting Training data")
        for token in tqdm(infos.keys()):
            frame_id = token.split('_')[3][:-4]
            # print(frame_id)
            record = {} # train data container for annotated frame
            info = infos[token]            
            # get point cloud
            pc_path = info['path']
            ref_obj = get_obj(hydra.utils.to_absolute_path(pc_path))
            pc = ref_obj['lidars']['points_xyz'] # get point cloud
            pc_feat = ref_obj['lidars']['points_feature'] # get point cloud features
            pc = np.concatenate((pc, pc_feat), axis = 1)
            # get transformation matrix
            anno_path = info['anno_path']
            ref_obj = get_obj(hydra.utils.to_absolute_path(anno_path))
            pose = np.reshape(ref_obj['veh_to_global'], [4, 4])
            # get ground truth ids for tracking
            num_points_in_gt = np.array([ann['num_points'] for ann in ref_obj['objects']])
            gt_global_ids = np.array([obj['name'] for obj in ref_obj['objects']])
            gt_boxes = np.array([ann['box'] for ann in ref_obj['objects']]).reshape(-1, 9)
            gt_names = np.array([self.TYPE_LIST[ann['label']] for ann in ref_obj['objects']])
            # remove empty boxes
            mask_not_zero = (num_points_in_gt >= 5).reshape(-1) #TODO: magic number !
            gt_boxes = gt_boxes[mask_not_zero, :].astype(np.float32) 
            gt_names = gt_names[mask_not_zero].astype(str)
            gt_global_ids = gt_global_ids[mask_not_zero].astype(str)
            num_points_in_gt = num_points_in_gt[mask_not_zero]
            assert gt_boxes.shape[0] == gt_names.shape[0]
            assert gt_global_ids.shape[0] == gt_names.shape[0]
            # mask whitelist
            mask_whitelist = np.array([True if label in self.WHITELIST else False for label in gt_names])
            gt_boxes = gt_boxes[mask_whitelist, :].astype(np.float32) 
            gt_names = gt_names[mask_whitelist].astype(str)
            gt_global_ids = gt_global_ids[mask_whitelist].astype(str)
            num_points_in_gt = num_points_in_gt[mask_whitelist]
            assert gt_boxes.shape[0] == gt_names.shape[0]
            assert gt_global_ids.shape[0] == gt_names.shape[0]
            assert gt_names.shape[0] == num_points_in_gt.shape[0]
            # create box_map
            boxes_map = {}
            for i in range(gt_boxes.shape[0]):
                boxes_map[gt_global_ids[i]] = gt_boxes[i]

            if gt_boxes.shape[0] == 0:
                log.debug("Empty scene, skip frame {}".format(frame_id))
                continue

            record['frame_id'] = frame_id
            record['point_cloud'] = pc
            record['boxes3d'] = gt_boxes
            record['boxes_id'] = gt_global_ids
            record['box_labels'] = gt_names
            record['pose'] = pose # use transorm_box function
            record['boxes_id_map'] = boxes_map # search boxes by unique id
            record['num_points_in_gt'] = num_points_in_gt #TODO: for debugging, remove later
            
            self.train_data_extract.append(record)
    
    def __len__(self):

        return len(self.train_data_extract) - 1 # avoid getting the last frame
    
    def __getitem__(self, idx):
        log.debug("Processing data extract {} and {}".format(idx, idx+1))
        # prepare detection data
        det_data = self.train_data_extract[idx+1]
        # if int(det_data['frame_id']) == 0: # skip the first frame in every sequence from matching with the last frame from the previous sequence !
        #     continue
        N = det_data['boxes3d'].shape[0]
        det_pc_in_box = []
        for i in range(N): # num of boxes
            corners = get_corners_from_labels_array(det_data['boxes3d'][i])
            pc_in_box, _ = extract_pc_in_box3d(det_data['point_cloud'], corners.T)
            if pc_in_box.shape[0] == 0:
                # Debugging for Visualizations
                from utils.visualizations import write_points_ply_file, write_oriented_bbox
                write_points_ply_file(det_data['point_cloud'][:,:3], 'debug.ply')
                write_oriented_bbox(det_data['boxes3d'][i].reshape(1,9), "debug_bboxs.ply")
                raise ValueError("empty boxes have been already filtered before !")        
                det['pc_in_box'] = np.zeros((1024,5)) #TODO: magic number !
            else:
                # subsample the points to a fixed size
                curr_det_pc_in_box = random_sampling(pc_in_box, 1024)
                det_pc_in_box.append(curr_det_pc_in_box)
        
        track_data = self.train_data_extract[idx]

        # prepare tracking data
        M = track_data['boxes3d'].shape[0]
        if N == 0 or M == 0: # Graph with no connections ! cannot consider this case in training !
            raise ValueError("should be handled in the init function of the dataset")
        track_pc_in_box = []
        for idx in range(M): # num of boxes
            corners = get_corners_from_labels_array(track_data['boxes3d'][idx])
            pc_in_box, _ = extract_pc_in_box3d(track_data['point_cloud'], corners.T)
            if pc_in_box.shape[0] == 0:
                print(pc_in_box.shape)
                from utils.visualizations import write_points_ply_file, write_oriented_bbox
                write_points_ply_file(track_data['point_cloud'][:,:3], 'debug.ply')
                
                write_oriented_bbox(track_data['boxes3d'][idx].reshape(1,9), "debug_bboxs.ply")
                raise ValueError("empty boxes have been already filtered before !")        
                det['pc_in_box'] = np.zeros((1024,5)) #TODO: magic number !
            else:
                # subsample the points to a fixed size
                curr_track_pc_in_box = random_sampling(pc_in_box, 1024) #TODO: magic number !
                track_pc_in_box.append(curr_track_pc_in_box)
        # LSTM input
        earliest_frame = int(track_data['frame_id']) - 5
        earliest_frame = max(-1, earliest_frame)
        track_boxes3d = torch.zeros(M,5,9) # last five tracks
        track_boxes3d[...] = torch.from_numpy(track_data['boxes3d']).unsqueeze(1) # add latest tracks and repeat them
        counters = [-1]*M # counter for every box
        for t in range(int(track_data['frame_id'])-1,earliest_frame,-1):
            for idx in range(M):
                curr_box_id = track_data['boxes_id'][idx]
                if curr_box_id in self.train_data_extract[t]['boxes_id_map'].keys():
                    # update and repeat, increase counter
                    # try:
                    track_boxes3d[idx, 0:counters[idx],:] = torch.from_numpy(self.train_data_extract[t]['boxes_id_map'][curr_box_id])
                    counters[idx] -= 1
        max_diff = np.array([self.max_dist[label] for label in det_data['box_labels']], np.float32)
        # construct diff matrix based on the glocal coordinates, by this you consider the true difference between the objects in different frames, if you calculate in the local coordinates of the frame, the difference between objects will be dependent on the speed of the car !
        # max distance in local coordinates
        # tracks_xy = track_data['boxes3d'][:, [0,1]].reshape(1,-1,2)
        # dets_xy = det_data['boxes3d'][:, [0,1]].reshape(-1,1,2)
        # dist = (((tracks_xy.reshape(1, -1, 2) - dets_xy.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
        # max distance in global coordinates
        tracks_xy_global = transform_box(track_data['boxes3d'], track_data['pose'])[:,[0,1]].reshape(1,-1,2)
        dets_xy_global = transform_box(det_data['boxes3d'], det_data['pose'])[:, [0,1]].reshape(-1,1,2)
        dist = (((tracks_xy_global.reshape(1, -1, 2) - dets_xy_global.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M

        assert dist.shape == (N,M)
        dist = np.sqrt(dist) # absolute distance in meter
        # invalid links
        adj_matrix = ((dist > max_diff.reshape(N, 1)) + (det_data['box_labels'].reshape(N, 1) != track_data['box_labels'].reshape(1, M))) > 0
        # log.debug(adj_matrix)
        adj_matrix =  ~np.array(adj_matrix) #valid links
        graph_adj_matrix = np.zeros((N+M, N+M)) # det first then tracks
        graph_adj_matrix[0:N,N::] = adj_matrix
        graph_adj_matrix[N::,0:N] = adj_matrix.T
        
        # ===== model forward + loss ====== #
        # transform_box(det_data['boxes3d'], det_data['pose']) try global box data # TODO
        # transform_box(track_data['boxes3d'], det_data['pose']) try global box data # TODO

        
        gt_affinity_matrix = det_data['boxes_id'].reshape(N, 1) == track_data['boxes_id'].reshape(1,M)
        gt_affinity_matrix = gt_affinity_matrix.astype(np.int)
        gt_affinity_matrix = torch.from_numpy(gt_affinity_matrix)

        # log.debug(gt_affinity_matrix)
        # log.debug(torch.from_numpy(adj_matrix.astype(int)))
        # log.debug(np.multiply(gt_affinity_matrix, adj_matrix.astype(int)))

        # make sure adj matrix covers all gt connections in gt_affinity_matrix
        assert adj_matrix.shape == gt_affinity_matrix.shape
        # log.debug(adj_matrix.astype(int))
        # log.debug(gt_affinity_matrix)
        # try:
        assert np.array_equal(torch.mul(gt_affinity_matrix, torch.from_numpy(adj_matrix.astype(int))), gt_affinity_matrix) == True
        # except:
        #     log.debug("create visualizations for debuggings")
        # from utils.visualizations import write_points_ply_file, write_oriented_bbox
        # write_points_ply_file(det_data['point_cloud'][:,:3], 'det_debug_{}.ply'.format(det_data['frame_id']))
        # write_points_ply_file(track_data['point_cloud'][:,:3], 'track_debug_{}.ply'.format(track_data['frame_id']))
        # write_oriented_bbox(det_data['boxes3d'].reshape(-1,9), "det_debug_bboxs_{}.ply".format(det_data['frame_id']))
        # write_oriented_bbox(track_data['boxes3d'].reshape(-1,9), "track_debug_bboxs_{}.ply".format(track_data['frame_id']))
        #     log.debug(adj_matrix.astype(int))
        #     log.debug(gt_affinity_matrix)
        #     raise ValueError("Error raised from tracking id: {} and det id {}".format(track_data['frame_id'], det_data['frame_id']))
        
        det_pc_in_box = torch.tensor(det_pc_in_box)
        assert det_pc_in_box.shape == torch.Size([N, 1024,5])
        track_pc_in_box = torch.tensor(track_pc_in_box)
        assert track_pc_in_box.shape == torch.Size([M, 1024,5])



        data_bundle = {
            # should return the data ready for the model
            'det_pc_in_box': det_pc_in_box, 
            'det_boxes3d': torch.from_numpy(det_data['boxes3d']).float(),
            'track_pc_in_box': track_pc_in_box, 
            'track_boxes3d': track_boxes3d, 
            'graph_adj_matrix': torch.from_numpy(graph_adj_matrix), # type: numpy
            'gt_affinity_matrix': gt_affinity_matrix,
        }
        return data_bundle