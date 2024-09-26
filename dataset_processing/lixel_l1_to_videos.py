import os
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
import shutil
import argparse
from typing import List
from concurrent.futures import ThreadPoolExecutor, Future

from tqdm import tqdm

DATA_DIR = os.path.join(WORKSPACE, 'data')
LIXEL_L1_DIR = os.path.join(DATA_DIR, 'lixel_l1')

def find_subdirs_names(
    url:str,
    filt_processed:bool) -> List[str]:
    subdirs_names = []
    for subdir_name in os.listdir(url):
        subdir_path = os.path.join(url, subdir_name)
        if os.path.isdir(subdir_path):
            video_dir_path = os.path.join(subdir_path, 'videos')
            if filt_processed and os.path.exists(video_dir_path):
                continue
            subdirs_names.append(subdir_name)
    return subdirs_names

def process_subdir(input_subdir:str) -> None:
    # copy images
    input_rgb_dir = os.path.join(input_subdir, 'reconstruct', 'images')
    video_dir = os.path.join(input_subdir, 'videos')
    for input_rgb_name in os.listdir(input_rgb_dir):
        _, _, rgb_direction = input_rgb_name.split('.')[0].split('_')
        video_subdir = os.path.join(video_dir, rgb_direction, 'images')
        os.makedirs(video_subdir, exist_ok=True)
        video_mask_subdir = os.path.join(video_dir, rgb_direction, 'masks')
        os.makedirs(video_mask_subdir, exist_ok=True)
        output_rgb_path = os.path.join(video_subdir, input_rgb_name)
        input_rgb_path = os.path.join(input_rgb_dir, input_rgb_name)
        rel_input_rgb_path = os.path.relpath(input_rgb_path, video_subdir)
        os.symlink(rel_input_rgb_path, output_rgb_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert lixel l1 data to 2dgs data')
    parser.add_argument('--lixel_l1_dir', type=str, default=LIXEL_L1_DIR, help='Path to active gauss data')
    parser.add_argument('--filt_processed', type=bool, default=True, help='Filter out processed data')
    
    args = parser.parse_args()
    lixel_l1_dir = args.lixel_l1_dir
    filt_processed = args.filt_processed
    
    subdirs_names = find_subdirs_names(lixel_l1_dir, filt_processed)
    
    progress_bar = tqdm(total=len(subdirs_names))
    
    futures:List[Future] = []
    with ThreadPoolExecutor() as executor:
        for subdir_name in subdirs_names:
            subdir_path = os.path.join(lixel_l1_dir, subdir_name)
            future = executor.submit(process_subdir, subdir_path)
            future.add_done_callback(lambda _: progress_bar.update(1))
            futures.append(future)
        for future in futures:
            future.result()