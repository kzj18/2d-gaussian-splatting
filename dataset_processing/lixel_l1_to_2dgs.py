import os
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
import sys
sys.path.append(WORKSPACE)
import time
import json
import shutil
import argparse
from typing import List
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import laspy
from tqdm import tqdm

from dataset_processing import find_subdirs_names

DATA_DIR = os.path.join(WORKSPACE, 'data')
LIXEL_L1_DIR = os.path.join(DATA_DIR, 'lixel_l1')
COLMAP_DIR = os.path.join(DATA_DIR, 'colmap')

HEIGHT = 1920
WIDTH = 1920

def process_subdir(
    subdir_name:str,
    input_dir:str,
    output_dir:str) -> None:
    global HEIGHT, WIDTH
    input_subdir = os.path.join(input_dir, subdir_name)
    output_subdir = os.path.join(output_dir, subdir_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    # copy images
    input_rgb_dir = os.path.join(input_subdir, 'reconstruct', 'images')
    output_rgb_dir = os.path.join(output_subdir, 'images')
    if os.path.exists(output_rgb_dir):
        shutil.rmtree(output_rgb_dir)
    shutil.copytree(input_rgb_dir, output_rgb_dir)
    
    # copy masks
    output_mask_dir = os.path.join(output_subdir, 'masks')
    os.makedirs(output_mask_dir, exist_ok=True)
    for i in range(6):
        input_mask_dir = os.path.join(input_subdir, 'videos', str(i), 'masks')
        for mask_name in os.listdir(input_mask_dir):
            input_mask_path = os.path.join(input_mask_dir, mask_name)
            output_mask_path = os.path.join(output_mask_dir, mask_name)
            shutil.copy(input_mask_path, output_mask_path)
    
    # load l3i.json
    transform_json_path = os.path.join(input_subdir, 'lidar_map', 'l3i.json')
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
        w = WIDTH
        h = HEIGHT
        fl_x = transform_json['camera']['fx_']
        fl_y = transform_json['camera']['fy_']
        cx = transform_json['camera']['cx_']
        cy = transform_json['camera']['cy_']
        cam_writer = f'1 PINHOLE {w} {h} {fl_x} {fl_y} {cx} {cy}'
        file.write(cam_writer)
    print(f"cameras data for {subdir_name} have been written to {cameras_path}")
    
    # write images.txt
    images_path = os.path.join(sparse0_dir, 'images.txt')
    with open(images_path, 'w') as file:
        image_index = 0
        for image_data in transform_json['images_data']:
            image_name:str = image_data['name']
            mask_path = os.path.join(output_mask_dir, image_name)
            if not os.path.exists(mask_path):
                continue
            qw = image_data['qw']
            qx = image_data['qx']
            qy = image_data['qy']
            qz = image_data['qz']
            tx = image_data['tx']
            ty = image_data['ty']
            tz = image_data['tz']
            traj_line = str(image_index) + ' ' +str(qw) + ' ' +str(qx) + ' ' +str(qy) + ' ' +str(qz) + ' ' +str(tx) + ' ' +str(ty) + ' ' +str(tz) + ' ' + str(1) +' '+ image_name +'\n\n'
            file.write(traj_line)
            image_index += 1
    print(f"traj data for {subdir_name} have been written to {images_path}")
    
    # write points3D.txt
    las = laspy.read(os.path.join(input_subdir, 'lidar_map', 'color_map_online_instan360_sfm_opt.las'))
    positions:np.ndarray = np.vstack((las.x, las.y, las.z)).T
    colors:np.ndarray = np.vstack((las.red, las.green, las.blue)).T
    assert colors.max() <= 255, "colors should be in range [0, 255]"
    points3d_path = os.path.join(sparse0_dir, 'points3D.txt')
    with open(points3d_path, 'w') as file:
        assert positions.shape[0] == colors.shape[0], "means3D and rgb_colors are not the same length."
        for point_id, (xyz, rgb) in enumerate(zip(positions, colors)):
            line = f"{point_id}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\t{rgb[0]}\t{rgb[1]}\t{rgb[2]}\n"
            file.write(line)
    print(f"point data for {subdir_name} have been written to {points3d_path}")
    
    with open(os.path.join(input_subdir, 'processed'), 'w') as file:
        file.write(f'Processed at {time.strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert lixel l1 data to 2dgs data')
    parser.add_argument('--lixel_l1_dir', type=str, default=LIXEL_L1_DIR, help='Path to active gauss data')
    parser.add_argument('--output_dir', type=str, default=COLMAP_DIR, help='Path to output data')
    parser.add_argument('--filt_processed', type=bool, default=True, help='Filter out processed data')
    parser.add_argument('--height', type=int, default=HEIGHT, help='Height of images')
    parser.add_argument('--width', type=int, default=WIDTH, help='Width of images')
    
    args = parser.parse_args()
    lixel_l1_dir = args.lixel_l1_dir
    output_dir = args.output_dir
    filt_processed = args.filt_processed
    
    subdirs_names = find_subdirs_names(lixel_l1_dir, filt_processed)
    
    progress_bar = tqdm(total=len(subdirs_names))
    
    futures:List[Future] = []
    with ThreadPoolExecutor() as executor:
        for subdir_name in subdirs_names:
            future = executor.submit(process_subdir, subdir_name, lixel_l1_dir, output_dir)
            future.add_done_callback(lambda _: progress_bar.update(1))
            futures.append(future)
        for future in futures:
            future.result()