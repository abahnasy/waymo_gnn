import numpy as np
from pyquaternion import Quaternion
import pickle
import copy
from tqdm import tqdm


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def transform_box(box, pose):
    """Transforms 3d upright boxes from one frame to another.
    Args:
    box: [..., N, 7] boxes.
    from_frame_pose: [...,4, 4] origin frame poses.
    to_frame_pose: [...,4, 4] target frame poses.
    Returns:
    Transformed boxes of shape [..., N, 7] with the same type as box.
    """
    transform = pose 
    heading = box[..., -1] + np.arctan2(transform[..., 1, 0], transform[..., 0,
                                                                    0])
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    box[..., 0:3]) + np.expand_dims(
                        transform[..., 0:3, 3], axis=-2)

    velocity = box[..., [6, 7]] 

    velocity = np.concatenate([velocity, np.zeros((velocity.shape[0], 1))], axis=-1) # add z velocity

    velocity = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    velocity)[..., [0, 1]] # remove z axis 

    return np.concatenate([center, box[..., 3:6], velocity, heading[..., np.newaxis]], axis=-1)


def veh_pos_to_transform(veh_pos):
    "convert vehicle pose to two transformation matrix"
    rotation = veh_pos[:3, :3] 
    tran = veh_pos[:3, 3]

    global_from_car = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=False
    )

    car_from_global = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=True
    )

    return global_from_car, car_from_global


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

def reorganize_info(infos):
    new_info = {}

    for info in infos:
        token = info['token']
        new_info[token] = info

    return new_info 


def label_to_name(label):
    if label == 0:
        return "VEHICLE"
    elif label == 1 :
        return "PEDESTRIAN"
    elif label == 2:
        return "CYCLIST"
    else:
        raise NotImplemented()


def sort_detections(detections):
    indices = [] 

    for det in detections:
        f = det['token']
        seq_id = int(f.split("_")[1])
        frame_id= int(f.split("_")[3][:-4])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    detections = [detections[r] for r in rank]

    return detections


def convert_detection_to_global_box(detections, infos):
    """
    Args:
        detections: Detector output
        infos: ground truth info
    Returns:
        sorted_return_list: list contains sorted predictions in global coordinates and timestamp
        detection_results: detections dictionary with token as keys, instead of detections list for easy access when compared with gt
    """
    ret_list = [] 

    detection_results = {} # copy.deepcopy(detections)

    for token in tqdm(infos.keys()):
        detection = detections[token]
        detection_results[token] = copy.deepcopy(detection)

        info = infos[token]
        # pose = get_transform(info)
        anno_path = info['anno_path']
        ref_obj = get_obj(anno_path)
        pose = np.reshape(ref_obj['veh_to_global'], [4, 4])

        box3d = detection["box3d_lidar"].detach().clone().cpu().numpy() 
        labels = detection["label_preds"].detach().clone().cpu().numpy()
        scores = detection['scores'].detach().clone().cpu().numpy()
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box3d[:, [3, 4]] = box3d[:, [4, 3]]

        box3d = transform_box(box3d, pose)

        frame_id = token.split('_')[3][:-4]

        num_box = len(box3d)

        anno_list =[]
        for i in range(num_box):
            anno = {
                'translation': box3d[i, :3],
                'velocity': box3d[i, [6, 7]],
                'detection_name': label_to_name(labels[i]),
                'score': scores[i], 
                'box_id': i 
            }

            anno_list.append(anno)

        ret_list.append({
            'token': token, 
            'frame_id':int(frame_id),
            'global_boxs': anno_list,
            'timestamp': info['timestamp'] 
        })

    sorted_ret_list = sort_detections(ret_list)

    return sorted_ret_list, detection_results 