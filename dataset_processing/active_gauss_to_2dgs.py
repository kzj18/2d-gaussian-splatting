import os
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
import sys
sys.path.append(WORKSPACE)
import time
import json
import shutil
import argparse
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from dataset_processing import find_subdirs_names

DATA_DIR = os.path.join(WORKSPACE, 'data')
ACTIVE_GAUSS_DIR = os.path.join(DATA_DIR, 'active_gauss')
COLMAP_DIR = os.path.join(DATA_DIR, 'colmap')

def process_subdir(
    subdir_name:str,
    input_dir:str,
    output_dir:str) -> None:
    input_subdir = os.path.join(input_dir, subdir_name)
    output_subdir = os.path.join(output_dir, subdir_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    # copy rgb images
    input_rgb_dir = os.path.join(input_subdir, 'gaussians_data', 'rgb')
    output_rgb_dir = os.path.join(output_subdir, 'images')
    if os.path.exists(output_rgb_dir):
        shutil.rmtree(output_rgb_dir)
    shutil.copytree(input_rgb_dir, output_rgb_dir)
    
    # load params.npz and transforms.json
    param_path = os.path.join(input_subdir, 'gaussians_data', 'params.npz')
    param_dict:Dict[str, np.ndarray] = dict(np.load(param_path, allow_pickle=True))
    transform_json_path = os.path.join(input_subdir, 'gaussians_data', 'transforms.json')
    with open(transform_json_path, 'r') as file:
        transform_json = json.load(file)
    
    # mkdir sparse/0
    sparse_dir = os.path.join(output_subdir, 'sparse')
    os.makedirs(sparse_dir, exist_ok=True)
    sparse0_dir = os.path.join(sparse_dir, '0')
    os.makedirs(sparse0_dir, exist_ok=True)
    
    # write cameras.txt
    cameras_path = os.path.join(sparse0_dir, 'cameras.txt')
    with open(cameras_path, 'w') as file:
        w = transform_json['w']
        h = transform_json['h']
        fl_x = transform_json['fl_x']
        fl_y = transform_json['fl_y']
        cx = transform_json['cx']
        cy = transform_json['cy']
        cam_writer = f'1 PINHOLE {w} {h} {fl_x} {fl_y} {cx} {cy}'
        file.write(cam_writer)
    print(f"cameras data for {subdir_name} have been written to {cameras_path}")
    
    # write images.txt
    images_path = os.path.join(sparse0_dir, 'images.txt')
    with open(images_path, 'w') as file:
        ts:np.ndarray = param_dict['cam_trans'][0]
        ts = ts.transpose()
        rs:np.ndarray = param_dict['cam_unnorm_rots'][0]
        rs = rs.transpose()
        for i, r in enumerate(rs):
            q = Quaternion(r/np.linalg.norm(r))
            R = q.rotation_matrix
            T = np.zeros((4,4))
            T[:3,:3] = R
            T[:3,3] = ts[i]
            T[3,3] = 1
            traj_line = str(i) + ' ' +str(q.w) + ' ' +str(q.x) + ' ' +str(q.y) + ' ' +str(q.z) + ' ' +str(ts[i][0]) + ' ' +str(ts[i][1]) + ' ' +str(ts[i][2]) + ' ' + str(1) +' '+ str(i).zfill(4)+'.png\n\n'
            file.write(traj_line)
    print(f"traj data for {subdir_name} have been written to {images_path}")
    
    # write points3D.txt
    points3d_path = os.path.join(sparse0_dir, 'points3D.txt')
    with open(points3d_path, 'w') as file:
        means3D = param_dict['means3D']
        rgb_colors = param_dict['rgb_colors']
        assert means3D.shape[0] == rgb_colors.shape[0], "means3D and rgb_colors are not the same length."
        for point_id, (xyz, rgb) in enumerate(zip(means3D, rgb_colors)):
            normalized_rgb = np.clip(rgb, 0, 1)
            scaled_rgb = normalized_rgb * 255
            line = f"{point_id}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\t{int(scaled_rgb[0])}\t{int(scaled_rgb[1])}\t{int(scaled_rgb[2])}\n"
            file.write(line)
    print(f"point data for {subdir_name} have been written to {points3d_path}")
    
    with open(os.path.join(input_subdir, 'processed'), 'w') as file:
        file.write(f'Processed at {time.strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert active gauss data to 2dgs data')
    parser.add_argument('--active_gauss_dir', type=str, default=ACTIVE_GAUSS_DIR, help='Path to active gauss data')
    parser.add_argument('--output_dir', type=str, default=COLMAP_DIR, help='Path to output data')
    parser.add_argument('--filt_processed', type=bool, default=True, help='Filter out processed data')
    
    args = parser.parse_args()
    active_gauss_dir = args.active_gauss_dir
    output_dir = args.output_dir
    filt_processed = args.filt_processed
    
    subdirs_names = find_subdirs_names(active_gauss_dir, filt_processed)
    
    progress_bar = tqdm(total=len(subdirs_names))
    
    futures:List[Future] = []
    with ThreadPoolExecutor() as executor:
        for subdir_name in subdirs_names:
            future = executor.submit(process_subdir, subdir_name, active_gauss_dir, output_dir)
            future.add_done_callback(lambda _: progress_bar.update(1))
            futures.append(future)
        for future in futures:
            future.result()