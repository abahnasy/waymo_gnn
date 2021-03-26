""" Visualization helper functions for detection and tracking
"""


from waymo_dataset.pipelines.loading import LoadPointCloudAnnotations, LoadPointCloudFromFile
from waymo_dataset.waymo import WaymoDataset
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh
import numpy as np
from pyquaternion import Quaternion
import PIL
import io


def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """
    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[2,2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        lengths = box[3:6]
        ctr = box[0:3]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[8])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    # trimesh.io.export.export_mesh.export(mesh_list, out_filename, file_type='ply')
    mesh_list.export(out_filename, file_type='ply')
    
    return

# [box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, ref_velocity[0], ref_velocity[1], box.heading]
def get_corners_from_labels_array(label, wlh_factor: float = 1.0) -> np.ndarray:
    ''' takes 1x8 array contains label information
    Args:
        np.array 1x8 contains label information [x, y, z, l, w, h, heading, labels]
    Returns:
    '''
    length = label[3] * wlh_factor
    width = label[4] * wlh_factor
    height = label[5] * wlh_factor
    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))
    orientation = Quaternion(axis=(0.0, 0.0, 1.0), radians=label[8]) # heading angle
    # Rotate
    corners = np.dot(orientation.rotation_matrix, corners)
    # Translate
    x, y, z = label[0], label[1], label[2]
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z
    return corners

def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)


def draw_point_cloud(pc, bboxes, bboxes_names, ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
    """ Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    axes_limits = [
        [-80, 80], # X axis range
        [-80, 80], # Y axis range
        [-3, 10]   # Z axis range
    ]
    axes_str = ['X', 'Y', 'Z']
    colors = {
        "PEDESTRIAN": 'b',
        "VEHICLE": 'r',
        "CYCLIST": 'g',
        "SIGN": 'c',
        "NONE": 'm'
    }
    ax.scatter(*np.transpose(pc[:, axes]),s=0.02, cmap='gray')
    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)
    for i in range(bboxes.shape[0]):
        label_type = bboxes_names[i] # get label
        box_corners = get_corners_from_labels_array(bboxes[i])
        draw_box(ax, box_corners, axes=axes, color=colors[label_type])
    
def pc_bev(pc, bboxes, seq_no, frame_no):
    # Draw point cloud data as plane projections
    f, ax3 = plt.subplots(1, 1, figsize=(25, 25))
    draw_point_cloud(
        pc,
        bboxes,
        bboxes_names,
        ax3, 
        'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
        axes=[0, 1] # X and Y axes
    )
    # plt.savefig("./viz/bev_view.png_{}_{}.png".format(seq_no, frame_no))
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

    

load_pc_step = LoadPointCloudFromFile()
load_anno_step = LoadPointCloudAnnotations()
ds = WaymoDataset(
    info_path="data/Waymo/infos_train_01sweeps_filter_zero_gt.pkl",
    root_path="data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl",
    pipeline=[load_pc_step, load_anno_step]
    )
print(len(ds))
# print(ds[1]["lidar"]["points"].shape)
points = ds[1]["lidar"]["points"] # [N,5]
# [box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, ref_velocity[0], ref_velocity[1], box.heading]
bboxes = ds[1]["lidar"]["annotations"]["boxes"]
bboxes_names = ds[1]["lidar"]["annotations"]["names"]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points[:,:3]) # extract xyz
# o3d.io.write_point_cloud("./viz/dummy.ply", pcd)
# pc_bev(points[:,:3], bboxes)
def buffer_plot_and_get():
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

seq = 1
images = []
c = 0
for frame in ds:
    seq_no = int(frame['metadata']['token'].split('_')[1])
    print(seq_no, " ", seq)
    
    if seq_no == seq:
        frame_no = frame['metadata']['token'].split('_')[3].split('.')[0]
        points = frame["lidar"]["points"][:,:3] # [N,3]
        bboxes = frame["lidar"]["annotations"]["boxes"]
        bboxes_names = frame["lidar"]["annotations"]["names"]
        print("Processing seq {}, frame {}".format(seq_no, frame_no))
        images.append(pc_bev(points, bboxes, seq_no, frame_no))
        if c == 10:
            break
        else:
            c += 1

images[0].save('./viz/pillow_imagedraw.gif', save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
    
    






# write_oriented_bbox(bboxes, "./viz/bboxes.ply")
