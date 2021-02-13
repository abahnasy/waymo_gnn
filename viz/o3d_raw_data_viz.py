import pickle
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
# from mpl_toolkits import mplot3d


with open("./annos_seq_300_frame_25.pkl", 'rb') as f:
	annos = pickle.load(f)

with open("./lidar_seq_300_frame_25.pkl", 'rb') as f:
	lidar = pickle.load(f)

pcl = lidar['lidars']['points_xyz']


# extract format from raw data !
#'box': np.array([box.center_x, box.center_y, box.center_z,
 #                        box.length, box.width, box.height, ref_velocity[0], 
  #                       ref_velocity[1], box.heading], dtype=np.float32)





pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcl)
# o3d.visualization.draw_geometries([pcd]) # default viewer
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)







colors_palette = {0:np.array([0,0,0]), # TYPE_UNKNOWN = 0;
					1:np.array([255, 0, 0]), # TYPE_VEHICLE = 1;
					2:np.array([255, 128, 0]), # TYPE_PEDESTRIAN = 2;
					3:np.array([0, 0, 255]), # TYPE_SIGN = 3;
					4:np.array([0, 255, 0]) # TYPE_CYCLIST = 4;
					} 
for obj in annos['objects']:
	obj_label = obj['label']	
	bbox_color = colors_palette[obj_label]
	center = np.reshape(obj['box'][0:3], (3,1))
	rotm = Quaternion(axis=(0.0, 0.0, 1.0), radians=obj['box'][-1])
	extent = np.reshape(obj['box'][3:6], (3,1))
	assert center.shape == (3,1)
	assert extent.shape == (3,1)
	assert rotm.rotation_matrix.shape == (3,3)
	obbox = o3d.geometry.OrientedBoundingBox(center, rotm.rotation_matrix, extent)
	obbox.color = bbox_color.astype(np.float)/255.0
	vis.add_geometry(obbox)


opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
vis.run()
vis.destroy_window()





