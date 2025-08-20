import cv2
import json
import time
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Manager
import sys, os
import yaml
import argparse

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

from utils.utils import parse_args, save_json_entry, convert_json_line_to_general, find_optimal_thread_count, find_scenes, get_existing_video_ids

from utils.utils import find_scenes_new
from utils.utils import load_existing_ids_new
import shutil
from heapq import heappush, heappop
from typing import List, Dict
#


def create_thumbnail_grid(video_path, grid_size=(3, 3)): 
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_needed = grid_size[0] * grid_size[1]
    frame_indices = np.linspace(0, total_frames-1, frames_needed, endpoint=False, dtype=int)
        
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 180)) 
            frames.append(frame)
    cap.release()
        
    if not frames:
        return None
            
    grid = np.zeros((grid_size[0] * 180, grid_size[1] * 320, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        i, j = divmod(idx, grid_size[1])
        grid[i*180:(i+1)*180, j*320:(j+1)*320] = frame
        
    return grid

def calculate_total_mp4_size(folder_path):
    total_size = 0  

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mp4"): 
                file_path = os.path.join(root, file)  
                total_size += os.path.getsize(file_path)  

    total_size_mb = total_size / (1024 * 1024)
    return total_size_mb

def main(data, output_folder, json_path, use_fixed_duration, detector_type, detector_threshold): 

    for item in tqdm(data, desc="video clips"):
        try:
            video_path = item['video-path']
            audio_path = item['audio-path']
            video_id = item['video-id'] 
            subtitle_path = item['subtitle-path']

            json_path_clip = json_path[:-5] +"_Clip_Time_Size.json" 
            existing_ids = load_existing_ids_new(json_path_clip)
            if video_id in existing_ids:
                continue    
            ##

            if os.path.exists(video_path) and os.path.exists(audio_path):
                output_subfolder = os.path.join(output_folder, video_id)
                video_url = None

                scenes_data, start_end_time_of_scenes, flg = find_scenes_new(video_path, audio_path, output_subfolder, video_id, 
                                                                                       video_url, subtitle_path, use_fixed_duration, 
                                                                                       detector_type, detector_threshold)            
                
                for scene_data in scenes_data:
                    save_json_entry(scene_data, json_path)
                else:
                    print(f"Warning: No metadata found for video_id {video_id}")
            else:
                print(f"Warning: Missing video or audio file in {video_id}")
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

def delete_unused_folders(output_folder, valid_video_ids):
    try:
        for folder_name in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder_name)
            if os.path.isdir(folder_path) and folder_name not in valid_video_ids:
                print(f"Deleting folder and its contents: {folder_path}")
                shutil.rmtree(folder_path)  
    except Exception as e:
        print(f"Error deleting folders: {e}")


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Run video head filter with multiprocessing.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()
    
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)['rough_clip']

    detector_type =  config['detector_type']
    detector_threshold = config['detector_threshold']
        
    input_json_file = config['input_json_file']
    output_folder = config['video_clips_folder']

    output_json_folder = config['output_json_folder']
    json_path = config['json_path']
    # Ensure the output_json_folder directory exists, create if not
    if not os.path.isdir(output_json_folder):
        os.makedirs(output_json_folder)

    data = json.load(open(input_json_file))

    json_path_clip = json_path[:-5] +"_Clip_Time_Size.json" 
    existing_ids = load_existing_ids_new(json_path_clip)
    data  = [video_data for video_data in data if video_data["video-id"] not in existing_ids] 
    print(len(data))
    delete_unused_folders(output_folder, existing_ids)

    part_number = 0

    use_fixed_duration = config['use_fixed_duration']

    num_samples = len(data)

    optimal_threads = find_optimal_thread_count(num_samples, config['max_threads'], config['threshold'])
    print("num_samples needed for process:", num_samples)
    print("num_samples per process:", optimal_threads)

    num_process = optimal_threads

    num_per_process = num_samples // num_process
    print("num_per_process", num_per_process)

    processes = []
    start_time = time.time()

    for idx in range(num_process):
        start_idx = idx * num_per_process
        end_idx = start_idx + num_per_process if idx < num_process - 1 else num_samples
        p_data = data[start_idx:end_idx]
        p = Process(target=main, args=(p_data, output_folder, json_path, use_fixed_duration, detector_type, detector_threshold))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_time_without_convert_json = time.time()

    convert_json_line_to_general(json_path) 


    end_time = time.time()

    print(f"{detector_type}Detector_threshold= {detector_threshold}")
    
    print("video clips time:", end_time - start_time)

    Total_time = {
            "detector_type:" : detector_type,
            "detector_threshold:" : detector_threshold,
            "total_time_without_convert_json:" : end_time_without_convert_json - start_time,
            "total_time": end_time - start_time
        }
    json_path_clip = json_path[:-5] +"_Clip_Time_Size.json"
    save_json_entry(Total_time, json_path_clip)
