import os
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
import json
import argparse
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

from tqdm import tqdm
import cv2
import numpy as np

DATA_DIR = os.path.join(WORKSPACE, 'data')
LIXEL_L1_DIR = os.path.join(DATA_DIR, 'lixel_l1')

def generate_tasks(
    input_subdir:str,
    filt_processed:bool) -> List[Tuple[str, str]]:
    # copy images
    tasks_list = []
    input_videos_dir = os.path.join(input_subdir, 'videos')
    if not os.path.exists(input_videos_dir):
        return tasks_list
    for i in range(6):
        input_video_dir = os.path.join(input_videos_dir, str(i))
        input_annotations_dir = os.path.join(input_video_dir, 'images')
        output_masks_dir = os.path.join(input_video_dir, 'masks')
        os.makedirs(output_masks_dir, exist_ok=True)
        file_names:List[str] = []
        for file_name in os.listdir(input_annotations_dir):
            file_url = os.path.join(input_annotations_dir, file_name)
            if os.path.isfile(file_url):
                file_names.append(file_name.split('.')[0])
        file_names = list(set(file_names))
        for file_name in file_names:
            input_image_path = os.path.join(input_annotations_dir, file_name + '.jpg')
            input_annotation_path = os.path.join(input_annotations_dir, file_name + '.json')
            output_mask_path = os.path.join(output_masks_dir, file_name + '.jpg')
            generate_task_flag = \
                os.path.exists(input_image_path) and \
                os.path.exists(input_annotation_path) and \
                ((not os.path.exists(output_mask_path)) or (not filt_processed))
            if generate_task_flag:
                tasks_list.append((input_annotation_path, output_mask_path))
    return tasks_list
        
def annotation_to_mask(
    input_annotation_path:str,
    output_mask_path:str) -> None:
    with open(input_annotation_path, 'r') as file:
        annotation = json.load(file)
    image_height = annotation['imageHeight']
    image_width = annotation['imageWidth']
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for shape in annotation['shapes']:
        polygon = np.array(shape['points']).astype(np.int32)
        mask = cv2.fillPoly(mask, [polygon], 255)
    cv2.imwrite(output_mask_path, mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert lixel l1 data to 2dgs data')
    parser.add_argument('--lixel_l1_dir', type=str, default=LIXEL_L1_DIR, help='Path to active gauss data')
    parser.add_argument('--filt_processed', type=bool, default=True, help='Filter out processed data')
    
    args = parser.parse_args()
    lixel_l1_dir = args.lixel_l1_dir
    filt_processed = args.filt_processed
    
    subdirs_paths = []
    for subdir_name in os.listdir(lixel_l1_dir):
        subdir_path = os.path.join(lixel_l1_dir, subdir_name)
        if os.path.isdir(subdir_path):
            subdirs_paths.append(subdir_path)
    
    tasks_list:List[Tuple[str, str, str]] = []
    for subdir_path in subdirs_paths:
        tasks_list.extend(generate_tasks(subdir_path, filt_processed))
        
    progress_bar = tqdm(total=len(tasks_list))
    futures:List[Future] = []
    with ThreadPoolExecutor() as executor:
        for task in tasks_list:
            future = executor.submit(annotation_to_mask, *task)
            future.add_done_callback(lambda _: progress_bar.update(1))
            futures.append(future)
        for future in futures:
            future.result()