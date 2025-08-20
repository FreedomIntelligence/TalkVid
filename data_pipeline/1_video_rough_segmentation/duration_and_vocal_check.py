import json
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import parse_args, save_json_entry, convert_json_line_to_general, find_optimal_thread_count
import re
import time
import cv2
import argparse
import yaml


    

def convert_time_to_seconds(time_str): 
    time_parts = time_str.split(':')
    
    if len(time_parts) == 3:
        hours, minutes, seconds = map(int, time_parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(time_parts) == 2:
        minutes, seconds = map(int, time_parts)
        return minutes * 60 + seconds
    else:
        raise ValueError(f"Unsupported time format: {time_str}")



def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file: {}".format(video_path))
    
    fps = cap.get(cv2.CAP_PROP_FPS)  
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  
    cap.release()
    
    if fps > 0:
        duration = frame_count / fps  
    else:
        duration = 0
    return duration


def duration_check(json_path, threshold):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    new_json_path = json_path.replace(".json", f"_duration_threshold={threshold}.json")
    for item in data:
        duration_str = item["durations"]
        duration = float(duration_str.replace("s", ""))
        if duration > threshold:
            save_json_entry(item, new_json_path)
    

    convert_json_line_to_general(new_json_path)
    return new_json_path
  

def load_subtitles(subtitle_path):
    """load subtitles from a .vtt file"""
    subtitles = []
    if not os.path.isdir(subtitle_path) and subtitle_path != '': 
        with open(subtitle_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            subtitle_blocks = re.findall(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})(.*?)\n\n', content, re.DOTALL)
            for start, end, text in subtitle_blocks:
                subtitles.append((start, end, text.strip()))
    return subtitles


def has_subtitle_in_range(subtitle_path, start_time, end_time):
    def convert_to_seconds(time_str):
        """convert subtitle time string to seconds"""
        time_parts = re.split('[:.]', time_str)
        hours = float(time_parts[0])
        minutes = float(time_parts[1])
        seconds = float(time_parts[2]) + float(time_parts[3]) / 1000 if len(time_parts) > 3 else float(time_parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def remove_html_tags(text):
        """remove HTML tags from text"""
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text

    subtitles = load_subtitles(subtitle_path)
    
    for start, end, text in subtitles:
        subtitle_start_time = convert_to_seconds(start)
        subtitle_end_time = convert_to_seconds(end)
        
        if subtitle_start_time <= end_time and subtitle_end_time >= start_time:
            # remove HTML tags
            clean_text = remove_html_tags(text)
            
            # remove "align:start position:0%" parts
            clean_text = re.sub(r'align:start position:0%', '', clean_text).strip()
            
            # judge if the subtitle is not empty and not '[Music]'
            if clean_text and clean_text != '[Music]':
                return True
    
    return False

def vocal_check(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    new_json_path = json_path.replace(".json", f"vocal_checked.json")
    for item in data:
        subtitle_path = item["subtitle_path"]
        if subtitle_path == "" and not os.path.isdir(subtitle_path):
            
            save_json_entry(item, new_json_path)
            continue
        start_time = item["start-time"]
        end_time = item["end-time"]
        if has_subtitle_in_range(subtitle_path, start_time, end_time):
            save_json_entry(item, new_json_path)
    
    convert_json_line_to_general(new_json_path)
    return new_json_path


def add_subtitle_path(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        json_path_with_subtitle = json_path.replace(".json", f"_subtitle_path_added.json")
        for item in data:
            if "subtitle_path" in item.keys():
                save_json_entry(item, json_path_with_subtitle)
                continue
            original_video = item["original-video"]
            folder_path = os.path.dirname(original_video)
            subtitle_extensions = ['.srt', '.vtt']
            subtitle_files = ""
            for file_name in os.listdir(folder_path):
                if any(file_name.endswith(ext) for ext in subtitle_extensions):
                    subtitle_files = os.path.join(folder_path, file_name)
                    break  # find the first subtitle file
            item["subtitle_path"] = subtitle_files
            save_json_entry(item, json_path_with_subtitle)
        convert_json_line_to_general(json_path_with_subtitle)
    return json_path_with_subtitle

        

def main():
    parser = argparse.ArgumentParser(description="Run video head filter with multiprocessing.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()
    
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)['rough_clip']

    detector_type =  config['detector_type']
    detector_threshold = config['detector_threshold']

    av1_flg = config['av1_flg']
    output_json_folder = config['output_json_folder'] + "_av1" if av1_flg else config['output_json_folder']
    start_sample = config['start_sample']
    end_sample = config['end_sample']

    json_path = os.path.join(output_json_folder,  f"_{detector_type}_threshold={detector_threshold}_{start_sample}_{end_sample}.json")

    if start_sample == -1:
        json_path = os.path.join(output_json_folder,  f"_{detector_type}_threshold={detector_threshold}_previous.json")
    duration_threshold = config['duration_threshold']
    

    new_json_path = duration_check(json_path, duration_threshold)
    new_json_path_vocal = vocal_check(new_json_path)




if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds.")
