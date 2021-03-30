""" Utility functions for visualizations
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


def write_points_ply_file(points, filename):
    assert points.shape[1] == 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) # extract xyz    
    o3d.io.write_point_cloud(filename, pcd)



def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 9 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
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
        # face_colors = [
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1],
        #     [255, 0, 0, 0.1]
        # ]
        box_trimesh_fmt = trimesh.creation.box(lengths, trns, face_colors=None)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    # trimesh.io.export.export_mesh.export(mesh_list, out_filename, file_type='ply')
    mesh_list.export(out_filename, file_type='ply')


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


def draw_point_cloud(pc, bboxes, bboxes_names, pred_bboxes, pred_names, ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
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
        # "SIGN": 'c',
        # "NONE": 'm'
    }
    pred_colors = {
        "PEDESTRIAN": 'y',
        "VEHICLE": 'c',
        "CYCLIST": 'm',
        # "SIGN": 'c',
        # "NONE": 'm'
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
    
    for i in range(pred_bboxes.shape[0]):
        label_type = pred_names[i] # get label
        box_corners = get_corners_from_labels_array(pred_bboxes[i])
        draw_box(ax, box_corners, axes=axes, color=pred_colors[label_type])

def create_bev_view(pc, bboxes, bboxes_names, pred_boxes, pred_names, filename):
    # Draw point cloud data as plane projections
    f, ax3 = plt.subplots(1, 1, figsize=(25, 25))
    draw_point_cloud(
        pc,
        bboxes,
        bboxes_names,
        pred_boxes, 
        pred_names,
        ax3, 
        'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
        axes=[0, 1] # X and Y axes
    )
    plt.savefig(filename)
    # buf = io.BytesIO()
    # plt.savefig(buf)
    # buf.seek(0)
    # return PIL.Image.open(buf)




