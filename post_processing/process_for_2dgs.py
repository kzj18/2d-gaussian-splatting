import os
import shutil
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import json

results_need_post_optim = os.listdir('results_need_post_optim')

results_subdir = [item for item in results_need_post_optim if os.path.isdir(os.path.join('results_need_post_optim', item))]

print(results_subdir)

data_for_2dgs_subdir = [name.split('_', 2)[-1] for name in results_subdir]

print(data_for_2dgs_subdir)

for item in data_for_2dgs_subdir:
    os.makedirs(os.path.join('data_for_2dgs', item), exist_ok=True)
    

for item in tqdm(results_subdir):
    # copy rgb images
    rgb_dir = os.path.join('results_need_post_optim', item, 'gaussians_data', 'rgb')
    target_dir_name = item.split('_', 2)[-1]
    target_dir = os.path.join('data_for_2dgs', target_dir_name, 'images')
    os.makedirs(target_dir, exist_ok=True)
    for file_name in tqdm(os.listdir(rgb_dir)):
        file_path = os.path.join(rgb_dir, file_name)
        if os.path.isfile(file_path):
            shutil.copy(file_path, target_dir)
            
    # load params.npz and transforms.json
    param_path = os.path.join('results_need_post_optim', item, 'gaussians_data', 'params.npz')
    param_dict = dict(np.load(param_path, allow_pickle=True))
    transform_json_path = os.path.join('results_need_post_optim', item, 'gaussians_data', 'transforms.json')
    
    # mkdir sparse/0
    sparse_dir = os.path.join('data_for_2dgs', target_dir_name, 'sparse')
    os.makedirs(sparse_dir, exist_ok=True)
    sparse0_dir = os.path.join(sparse_dir, '0')
    os.makedirs(sparse0_dir, exist_ok=True)
    
    # write cameras.txt
    cameras_path = os.path.join(sparse0_dir, 'cameras.txt')
    cameras_file = open(cameras_path, 'w')
    with open(transform_json_path, 'r') as file:
        transform_json = json.load(file)
    w = transform_json['w']
    h = transform_json['h']
    fl_x = transform_json['fl_x']
    fl_y = transform_json['fl_y']
    cx = transform_json['cx']
    cy = transform_json['cy']
    cam_writer = f'1 PINHOLE {w} {h} {fl_x} {fl_y} {cx} {cy}' 
    cameras_file.write(cam_writer)
    cameras_file.close()
    print(f"cameras data  for {item} have been writen to {cameras_path}")
    
    # write images.txt
    images_path = os.path.join(sparse0_dir, 'images.txt')
    traj_file = open(images_path, 'w')
    ts = param_dict['cam_trans'][0].transpose()
    rs = param_dict['cam_unnorm_rots'][0].transpose()
    for i, r in enumerate(rs):
        q = Quaternion(r/np.linalg.norm(r))
        R = q.rotation_matrix
        T = np.zeros((4,4))
        T[:3,:3] = R
        T[:3,3] = ts[i]
        T[3,3] = 1
        traj = np.squeeze(T.reshape(1,16))
        traj_line = str(i) + ' ' +str(q.w) + ' ' +str(q.x) + ' ' +str(q.y) + ' ' +str(q.z) + ' ' +str(ts[i][0]) + ' ' +str(ts[i][1]) + ' ' +str(ts[i][2]) + ' ' + str(1) +' '+ str(i).zfill(4)+'.png\n\n'
        traj_file.write(traj_line)
    traj_file.close()
    print(f"traj data for {item} have been writen to {images_path}")
    
    
    # wirte points3D.txt
    points3d_path = os.path.join(sparse0_dir, 'points3D.txt')
    points3d_file = open(points3d_path, 'w')
    means3D = param_dict['means3D']
    rgb_colors = param_dict['rgb_colors']
    assert means3D.shape[0] == rgb_colors.shape[0], "means3D and rgb_colors are not the same length."
    for point_id, (xyz, rgb) in enumerate(tqdm(zip(means3D, rgb_colors))):
        normalized_rgb = np.clip(rgb, 0, 1)
        scaled_rgb = normalized_rgb * 255
        line = f"{point_id}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\t{int(scaled_rgb[0])}\t{int(scaled_rgb[1])}\t{int(scaled_rgb[2])}\n"
        points3d_file.write(line)
    points3d_file.close()
    print(f"point data for {item} have been writen to {points3d_path}")