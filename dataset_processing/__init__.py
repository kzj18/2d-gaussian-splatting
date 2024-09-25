import os
from typing import List

def find_subdirs_names(
    url:str,
    filt_processed:bool) -> List[str]:
    subdirs_names = []
    for subdir_name in os.listdir(url):
        subdir_path = os.path.join(url, subdir_name)
        if os.path.isdir(subdir_path):
            processed_flag_url = os.path.join(subdir_path, 'processed')
            if filt_processed and os.path.exists(processed_flag_url):
                continue
            subdirs_names.append(subdir_name)
    return subdirs_names

