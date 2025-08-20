import os
import re
import sys
import argparse
from tqdm import tqdm
import cv2
from scenedetect import open_video, SceneManager, AdaptiveDetector, ContentDetector,ThresholdDetector,HistogramDetector
from scenedetect.stats_manager import StatsManager
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, AudioFileClip


import subprocess
import requests
from PIL import Image
from collections import Counter
import json
import math 
import munch
import yaml
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable as V
from torch.nn import functional as F


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

NUM_FRAMES_POINTS = {
    1: (0.5,),
    2: (0.25, 0.5),
    3: (0.1, 0.5, 0.9),
    9: (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
}


def save_max_file(max_ckpt_file, part_number):
    # Get the directory path from the file path
    directory = os.path.dirname(max_ckpt_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Write the max_number to the output file
    with open(max_ckpt_file, 'w') as f:
        f.write(str(part_number))


def read_max_number(file_path):
    '''
    Read the maximum number from a file.
    
    :param file_path: The path to the file containing the maximum number.
    :return: The maximum number as an integer, or None if an error occurs.
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the content, assuming it contains only the maximum number as a string
            max_number_str = file.read().strip()  # Remove any leading/trailing whitespace
            # Convert the string to an integer
            max_number = int(max_number_str)
            return max_number
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except ValueError:
        print(f"The file does not contain a valid integer.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def find_max_number(folder_path, file_path, perfix=None):
    '''
    find the maximum number in the filenames of Excel files within a folder
    '''
    max_number = None
    if os.path.exists(file_path):
        max_number = read_max_number(file_path)
    
    if max_number is None:
        max_number = 0
        pattern = re.compile(r'\d+')  # Regex pattern to match digits in the filenames

        # Iterate through all files in the specified folder
        for filename in os.listdir(folder_path):
            # if filename.endswith('.xlsx'):  # Check if the file is an Excel file
            if perfix is None:
                check_name = 'all_scenes_data_part'
            else:
                check_name = f'all_scenes_data_{perfix}_part'
            if filename.endswith('.csv') or (filename.startswith(check_name) and filename.endswith('.json')):
                match = pattern.search(filename)
                if match:
                    number = int(match.group(0))  # Convert the matched number to an integer
                    if number > max_number:
                        max_number = number

    return max_number  # Return the max file number


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    args = munch.munchify(yaml.safe_load(open(arg.config)))

    return args


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=default_dump)


def save_json_entry(entry, path):
    with open(path, 'a') as outfile:
        json.dump(entry, outfile, separators=(',', ':'), default=default_dump)
        outfile.write('\n')


def extract_valid_json_from_malformed_line(line):
    """
    Extract valid JSON from a malformed line by identifying the first '{' before the last '}'.
    """
    buffer = ""
    collected_data = []
    
    # Find the position of the last closing brace '}'
    last_closing_brace = line.rfind('}')
    
    if last_closing_brace != -1:
        # Find the position of the first opening brace '{' before the last closing brace
        first_opening_brace = line[:last_closing_brace].rfind('{')
        
        if first_opening_brace != -1:
            # Extract the substring between the first opening brace and the last closing brace
            buffer = line[first_opening_brace:last_closing_brace + 1]
            try:
                # Try to parse the extracted substring as JSON
                sample = json.loads(buffer)
                collected_data.append(sample)  # If parsing is successful, store the JSON object
            except json.JSONDecodeError:
                print("Failed to parse JSON, it might still be incomplete.")
            except Exception as e:
                print(f"An error occurred: {e}")
    
    # Return the valid JSON data if found, otherwise return None
    if collected_data:
        return collected_data[0]
    else:
        return None


def check_output_file(output_file):
    """
    Check the given file for existing JSON records, including malformed lines.
    Extract valid JSON from incomplete lines if possible and store unique 'id' fields.
    """
    if not os.path.isfile(output_file):
        return None

    existing_ids = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        try:
            for line in f:
                try:
                    # First attempt to parse the line directly
                    sample = json.loads(line)
                    existing_ids.add(sample['id'])
                except json.JSONDecodeError:
                    # If the line is malformed, attempt to extract a valid JSON fragment
                    sample = extract_valid_json_from_malformed_line(line)
                    if sample and 'id' in sample:
                        existing_ids.add(sample['id'])
                except Exception as e:
                    print(f"An error occurred: {e}")
        except json.JSONDecodeError as e:
            print("This filtering process encountered an issue with the JSON format.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    return existing_ids


def load_existing_ids(path):
    existing_ids = set()
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                for line in f:
                    try:
                        entry = json.loads(line)
                        existing_ids.add(entry.get("id"))
                    except json.JSONDecodeError:
                        # If the line is malformed, attempt to extract a valid JSON fragment
                        entry = extract_valid_json_from_malformed_line(line)
                        if entry and 'id' in entry:
                            existing_ids.add(entry['id'])    
                    except Exception as e:

                        print(f"An error structured: {e}")              
            except json.JSONDecodeError as e:
                # if the json format is constructed but not processed completed?
                print("this filtering process completed")
                exit()
            except Exception as e:
                print(f"An error occurred: {e}")
                exit()
    return existing_ids

def load_existing_ids_new(path):
    existing_ids = list()

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines() 

        for line in lines:  
            entry = json.loads(line.strip())  
            if isinstance(entry, list): 
                entry = entry[0]
            existing_ids.append(entry.get("video_id:"))



    return existing_ids



def extract_video_id(sample, pattern):
    '''extract video id'''
    match = re.search(pattern, sample)
    return match.group(1)


def check_clips_output_file(output_file):
    if not os.path.isfile(output_file):
        return None
    existing_ids = set()
    pattern = r'video(.+?)-'
    with open(output_file, 'r', encoding='utf-8') as f:
        try:
            for line in f:
                try:
                    sample = json.loads(line)
                    video_id = extract_video_id(sample['id'], pattern)
                    if video_id:
                        existing_ids.add(video_id)
                except json.JSONDecodeError:
                    # If the line is malformed, attempt to extract a valid JSON fragment
                    sample = extract_valid_json_from_malformed_line(line)
                    video_id = extract_video_id(sample['id'], pattern)
                    if video_id:
                        existing_ids.add(video_id)
                except Exception as e:
                    print(f"An error structured: {e}")
        except json.JSONDecodeError as e:
            # if the json format is constructed but not processed completed?
            print("this filtering process completed")
        except Exception as e:
            print(f"An error occurred: {e}")
            exit()
    return existing_ids


def is_video(filename):
    ext = os.path.splitext(filename)[-1].lower()
    return ext in VID_EXTENSIONS


def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)


def get_video_codec(video_path):
    # Use FFmpeg to get video information
    cmd = ['ffmpeg', '-i', video_path]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    output = result.stderr.decode('utf-8')  # Video information is in stderr

    # Use regular expression to find video codec information
    match = re.search(r"Video:\s(\w+)", output)
    if match:
        return match.group(1)
    return None


def find_optimal_thread_count(total_samples, max_threads, threshold):
    optimal_thread_count = 1  
    if total_samples <= threshold:
        print(f"The number of samples is too small, and the returned optimal_thread_count is threshold: {threshold}")
        return total_samples
    for thread_count in range(1, max_threads + 1):
        samples_per_thread = total_samples // thread_count
        remaining_samples = total_samples - samples_per_thread * thread_count

        if remaining_samples <= threshold:
            optimal_thread_count = thread_count 

    return optimal_thread_count


def split_and_average_max(video_motion_scores):
    total_frames = len(video_motion_scores)
    points = [0.25, 0.5, 0.75]
    separate_frame = [int(p * total_frames) for p in points]

    # split list
    sublists = [
        video_motion_scores[:separate_frame[0]],
        video_motion_scores[separate_frame[0]:separate_frame[1]],
        video_motion_scores[separate_frame[1]:separate_frame[2]],
        video_motion_scores[separate_frame[2]:]
    ]

    # caculate every part's average
    averages = [np.mean(sublist) for sublist in sublists if sublist]  # make sure that sublist is not empty
    
    # get the maximum of the average
    max_average = max(averages) if averages else 0.0
    
    return round(max_average, 5)


def convert_to_mp4(input_path):
    output_path = os.path.splitext(input_path)[0] + ".mp4"
    command = [
        'ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', '-vsync', '2', output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def save_frames_from_video(video_path, output_folder, scene_id):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"{scene_id}_frame{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()


def visualization_and_filtering(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        video_data = json.load(f)

    print("Length of the JSON:", len(video_data))

    filtered_data = [sample for sample in video_data if len(sample.get('keyframes idx', [])) >= 5]

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)

    print("Filtered data (first 3 samples):")
    print(filtered_data[:3])


def ffmpeg_get_timestamp(video_path, audio_path, video_id, video_url, pass_avi):

    # Open video file to get video information
    video = VideoFileClip(video_path)
    width, height = video.size
    fps = video.fps

    video_codec = get_video_codec(video_path)

    if video_codec == 'h264' or pass_avi is not True:
        # Use scenedetect for scene detection
        video = open_video(video_path)
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)

        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=1.7))

        scene_manager.detect_scenes(video=video)
        scene_list = scene_manager.get_scene_list()

        scenes = [[scene[0].get_frames(), scene[1].get_frames()] for _, scene in enumerate(scene_list)]
        # Write data to JSON file
        data = {
            "check_id": video_id,
            "height": height,
            "width": width,
            "fps": fps,
            "original-video": video_path,
            "original-audio": audio_path,
            "video_url": video_url,
            "video_codec": video_codec,
            "scenes": scenes
        }
    else:

        data = {
            "check_id": video_id,
            "height": height,
            "width": width,
            "fps": fps,
            "original-video": video_path,
            "original-audio": audio_path,
            "video_url": video_url,
            "video_codec": video_codec,
        }

    return data


def find_scenes(video_path, audio_path, output_subfolder, video_id, video_url, use_fixed_duration):
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)


    # Open video file to get video information
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_cap.release()

    # Use scenedetect for scene detection
    video = open_video(video_path)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    scene_manager.add_detector(ContentDetector(threshold=10.0))

    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    scenes_data = []
    scene_counter = 1  # Initialize scene counter
    print(scene_list)

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        duration = end_time - start_time

        # Discard segments shorter than 2 seconds
        if duration < 2.0:
            continue

        if use_fixed_duration:
            # Adjust segments to match the desired durations
            if 5.0 <= duration < 10.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 2.5
                end_time = middle + 2.5
            
            elif 10.0 <= duration < 15.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 5.0
                end_time = middle + 5.0
                
            elif 15.0 <= duration < 20.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 7.5
                end_time = middle + 7.5
            elif 20.0 <= duration < 25.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 10.0
                end_time = middle + 10.0
                
            elif duration >= 25.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 12.5
                end_time = middle + 12.5
        else:
            # Adjust start and end times by subtracting 0.1 seconds
            start_time = max(0, start_time - 0.1)
            end_time = max(0, end_time - 0.1)   

        output_video_filename = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}.mp4")
        output_audio_filename = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}.m4a")

        # Suppress output of ffmpeg_extract_subclip
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # Extract video clip
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_video_filename)

        # Extract audio clip
        ffmpeg_extract_subclip(audio_path, start_time, end_time, targetname=output_audio_filename)

        # Restore stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

        # Now let's save the first frame of the video scene
        scene_video_cap = cv2.VideoCapture(output_video_filename)
        success, first_frame = scene_video_cap.read()
        if success:
            first_frame_path = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}_first_frame.png")
            cv2.imwrite(first_frame_path, first_frame)  # Save the first frame image
            save_flag = True
        else:
            os.remove(output_video_filename)
            os.remove(output_audio_filename)
            save_flag = False
        scene_video_cap.release()

        # Save frames from the segmented video
        # save_frames_from_video(output_video_filename, output_subfolder, f"video{video_id}_scene{i + 1}")

        if save_flag:
            scenes_data.append({
                "id": f"video{video_id}-scene{scene_counter}",
                "video-path": output_video_filename,
                "audio-path": output_audio_filename,
                "first-frame-path": first_frame_path,  # Add the first frame path to the scene data
                "height": height,
                "width": width,
                "fps": fps,
                "start-time": start_time,
                "start-frame": scene[0].get_frames(),
                "end-time": end_time,
                "end-frame": scene[1].get_frames(),
                "durations": f"{round(end_time - start_time, 1)}s",
                "original-video": video_path,
                "original-audio": audio_path,
                "video_url": video_url,
            
            })
        
        scene_counter += 1  # Increment scene counter

    return scenes_data


def find_scenes_new_test_error_detect(video_path, audio_path, output_subfolder, video_id, video_url, use_fixed_duration, json_path, detector_type, detector_parameter, method_id):
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # Convert video and audio to MP4 format if necessary
    # video_path = convert_to_mp4(video_path)
    # audio_path = convert_to_mp4(audio_path)

    # Open video file to get video information
    path_error_video_info = json_path[:-5] + "_error_videos.json"
    print(path_error_video_info)
    try:
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

    except cv2.error as e:
        print(f"OpenCV error with {video_path}: {e}")
        error = {"video_id": video_id, "video_path": video_path, "error_type": "OpenCV error"}
        save_json_entry(error, path_error_video_info)
        # log.write(f"OpenCV error with {video_path}: {e}\n")
        # traceback.print_exc() 

    except ValueError as e:
        print(f"{e}")
        # log.write(f"{e}\n")
        error = {"video_id": video_id, "video_path": video_path, "error_type": "Video error"}
        save_json_entry(error, path_error_video_info)
        # traceback.print_exc() 

    except Exception as e:
        print(f"Unexpected error with {video_path}: {e}")
        # log.write(f"Unexpected error with {video_path}: {e}\n")
        error = {"video_id": video_id, "video_path": video_path, "error_type": "Unexpected error"}
        save_json_entry(error, path_error_video_info)
        # traceback.print_exc()  
    return

def is_av1_video(video_path):
    import ffmpeg
    try:
        
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='stream=codec_name')
        
        
        codec_name = probe['streams'][0]['codec_name']
        
        if codec_name.lower() == 'av1':
            return True
        else:
            return False
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e}")
        return False

def find_scenes_new(video_path, audio_path, output_subfolder, video_id, 
                    video_url, subtitle_path, use_fixed_duration, 
                    detector_type, detector_threshold, method_id = 12):
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)


    video_cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG) 

    if not video_cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_cap.release()

    # Use scenedetect for scene detection
    video = open_video(video_path)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    ### Choose different detectors
    
    threshold = float(detector_threshold)
    if detector_type == "Adaptive":
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold)) 

    elif detector_type == "Histogram":
        scene_manager.add_detector(HistogramDetector(threshold = threshold))
    
    elif detector_type == "Content":
        scene_manager.add_detector(ContentDetector(threshold = threshold))
    
    elif detector_type == "Hash":
        scene_manager.add_detector(HashDetector(threshold = threshold))
    
    elif detector_type == "Threshold":
        scene_manager.add_detector(ThresholdDetector(threshold = threshold))

    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    scenes_data = []
    scene_counter = 1  # Initialize scene counter

    
    save_path_scenes_info = os.path.join(output_subfolder, f"0000_video{video_id}.txt")
     #
    start_end_time_of_scenes = ""

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds() + 0.4 
        end_time = scene[1].get_seconds()
        duration = end_time - start_time


        if use_fixed_duration:
            # Adjust segments to match the desired durations
            if 5.0 <= duration < 10.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 2.5
                end_time = middle + 2.5
            
            elif 10.0 <= duration < 15.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 5.0
                end_time = middle + 5.0
                
            elif 15.0 <= duration < 20.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 7.5
                end_time = middle + 7.5
            elif 20.0 <= duration < 25.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 10.0
                end_time = middle + 10.0
                
            elif duration >= 25.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 12.5
                end_time = middle + 12.5
        else:
            # Adjust start and end times by subtracting 0.1 seconds
            start_time = max(0, start_time - 0.1)
            end_time = max(0, end_time - 0.1)   

        output_video_filename = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}.mp4")
        output_audio_filename = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}.m4a")

        ## Debugging
        start_minutes, start_seconds = divmod(start_time, 60)
        end_minutes, end_seconds = divmod(end_time, 60)

        scene_info = (
            "scene " + str(scene_counter) +
            " infos: start_time " + str(int(start_minutes)) + ":" + str(int(start_seconds)) +
            ", end_time " + str(int(end_minutes)) + ":" + str(int(end_seconds)) + "\n"
        )

        start_end_time_of_scenes += scene_info
        os.makedirs(output_subfolder, exist_ok=True)

        
        with open(save_path_scenes_info, "a") as file: 
            file.write(f"scene {scene_counter} infos: start_time {int(start_minutes)}:{int(start_seconds)}, end_time {int(end_minutes)}:{int(end_seconds)}\n")
        #

        ############## Save the video and audio clips 
        if method_id == -1: # skip the clip of videos and audios
            continue

        # ######################################### 
        if method_id == 12:
            subprocess.run([
                "ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", str(end_time - start_time),
                "-c:v", "libx264", "-preset", "medium", "-an", output_video_filename
            ])

            subprocess.run([ 
                "ffmpeg", "-ss", str(start_time), "-i", audio_path, "-t", str(end_time - start_time),
                "-c:a", "aac", output_audio_filename
            ])
        ##############################################

        

        # Now let's save the first frame of the video scene
        scene_video_cap = cv2.VideoCapture(output_video_filename)
        success, first_frame = scene_video_cap.read()
        if success:
            first_frame_path = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}_first_frame.png")
            cv2.imwrite(first_frame_path, first_frame)  # Save the first frame image
            save_flag = True
        else:
            os.remove(output_video_filename)
            os.remove(output_audio_filename)
            save_flag = False
        scene_video_cap.release()

        # Save frames from the segmented video
        # save_frames_from_video(output_video_filename, output_subfolder, f"video{video_id}_scene{i + 1}")

        if save_flag:
            scenes_data.append({
                "id": f"video{video_id}-scene{scene_counter}",
                "video-path": output_video_filename,
                "audio-path": output_audio_filename,
                "first-frame-path": first_frame_path,  # Add the first frame path to the scene data
                "height": height,
                "width": width,
                "fps": fps,
                "start-time": start_time,
                "start-frame": scene[0].get_frames(),

                "end-time": end_time,
                "end-frame": scene[1].get_frames(),
                "durations": f"{round(end_time - start_time, 1)}s",
                "original-video": str(audio_path).replace(".m4a", ".mp4"), #using original audio path to get original video path
                "original-audio": audio_path,
                "video_url": video_url,
                "subtitle_path": subtitle_path,
            
            })
        
        scene_counter += 1  # Increment scene counter

    return scenes_data, start_end_time_of_scenes, True


def find_scenes_new_fixed_duration(item, video_path, audio_path, output_subfolder, video_id, 
                                    video_url, subtitle_path, use_fixed_duration, fixed_duration,
                                    method_id = 12):
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    
    video_cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG) 

    if not video_cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)  
    video_cap.release()

    video_duration = frame_count / fps  

    scenes_data = []
    scene_counter = 1  # Initialize scene counter

    save_path_scenes_info = os.path.join(output_subfolder, f"0000_video{video_id}.txt")
    start_end_time_of_scenes = ""

    scene_list = []
    start = 0
    
    while start + fixed_duration <= video_duration:
        end = start + fixed_duration
        scene_list.append((start, end))
        start = end  
    
   
    if start < video_duration:
        scene_list.append((video_duration - fixed_duration, video_duration))

    for i, scene in enumerate(scene_list):
        start_time = scene[0] 
        end_time = scene[1]

        output_video_filename = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}.mp4")
        output_audio_filename = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}.m4a")

        start_minutes, start_seconds = divmod(start_time, 60)
        end_minutes, end_seconds = divmod(end_time, 60)

        scene_info = (
            "scene " + str(scene_counter) +
            " infos: start_time " + str(int(start_minutes)) + ":" + str(int(start_seconds)) +
            ", end_time " + str(int(end_minutes)) + ":" + str(int(end_seconds)) + "\n"
        )

        start_end_time_of_scenes += scene_info
        
        os.makedirs(output_subfolder, exist_ok=True)

        
        with open(save_path_scenes_info, "a") as file:  
            file.write(f"scene {scene_counter} infos: start_time {int(start_minutes)}:{int(start_seconds)}, end_time {int(end_minutes)}:{int(end_seconds)}\n")
        #

        ############## Save the video and audio clips 
        if method_id == -1: # skip the clip of videos and audios
            continue

        # ######################################### 
        if method_id == 12:
            subprocess.run([
                "ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", str(end_time - start_time),
                "-c:v", "libx264", "-preset", "medium", "-an", output_video_filename
            ])

            subprocess.run([ 
                "ffmpeg", "-ss", str(start_time), "-i", audio_path, "-t", str(end_time - start_time),
                "-c:a", "aac", output_audio_filename
            ])
        ##############################################

        

        # Now let's save the first frame of the video scene
        scene_video_cap = cv2.VideoCapture(output_video_filename)
        success, first_frame = scene_video_cap.read()
        if success:
            first_frame_path = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}_first_frame.png")
            cv2.imwrite(first_frame_path, first_frame)  # Save the first frame image
            save_flag = True
        else:
            os.remove(output_video_filename)
            os.remove(output_audio_filename)
            save_flag = False
        scene_video_cap.release()


        if save_flag:
            son_item = item
            son_item.update({
                "id": f"video{video_id}-scene{scene_counter}",
                "video-path": output_video_filename,
                "audio-path": output_audio_filename,
                "first-frame-path": first_frame_path,  # Add the first frame path to the scene data
                "height": height,
                "width": width,
                "fps": fps,
                "start-time": start_time,

                "end-time": end_time,
                "durations": f"{round(end_time - start_time, 1)}s",
                "original-video": video_path,
                "original-audio": audio_path,
                "video_url": video_url,
                "subtitle_path": subtitle_path,
            
            })
            scenes_data.append(son_item.copy())
        
        scene_counter += 1  # Increment scene counter

    return scenes_data, start_end_time_of_scenes, True


import cv2

def cut_video_segment(video_path, start_frame, end_frame, output_path):
    """
    Cuts a video segment from start_frame to end_frame and saves it to output_path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Validate start and end frames
    if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
        cap.release()
        raise ValueError("Invalid start or end frame")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create output video file")

    # Quickly jump to the start frame (using grab for acceleration)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_frame = 0
    while current_frame < start_frame:
        if not cap.grab():
            cap.release()
            writer.release()
            raise RuntimeError("Failed to seek in video")
        current_frame += 1

    # Read and write frames from start_frame to end_frame
    for _ in range(int(end_frame - start_frame) + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    # Release resources
    cap.release()
    writer.release()
    return True


def find_scenes_new_fixed_frames(item, video_path, audio_path, output_subfolder, video_id, 
                                    video_url, subtitle_path, use_fixed_frames, fixed_frame,
                                    method_id = 'frame'):
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    
    video_cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG) 

    if not video_cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)  
    video_cap.release()

    video_duration = frame_count / fps  

    scenes_data = []
    scene_counter = 1  # Initialize scene counter

    save_path_scenes_info = os.path.join(output_subfolder, f"0000_video{video_id}.txt")
    start_end_time_of_scenes = ""

    scene_list = []
    start = 0
    
    while start + fixed_frame <= frame_count:
        end = start + fixed_frame
        scene_list.append((start, end))
        start = end  
    
    if start < frame_count:
        scene_list.append((frame_count - fixed_frame, frame_count))

    for i, scene in enumerate(scene_list):
        start_frame = scene[0] 
        end_frame = scene[1]

        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time
            
        output_video_filename = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}.mp4")
        output_audio_filename = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}.m4a")

        start_minutes, start_seconds = divmod(start_time, 60)
        end_minutes, end_seconds = divmod(end_time, 60)
    

        scene_info = (
            "scene " + str(scene_counter) +
            " infos: start_time " + str(int(start_minutes)) + ":" + str(int(start_seconds)) +
            ", end_time " + str(int(end_minutes)) + ":" + str(int(end_seconds)) + "\n"
        )

        start_end_time_of_scenes += scene_info

        os.makedirs(output_subfolder, exist_ok=True)

       
        with open(save_path_scenes_info, "a") as file:  
            file.write(f"scene {scene_counter} infos: start_time {int(start_minutes)}:{int(start_seconds)}, end_time {int(end_minutes)}:{int(end_seconds)}\n")
        #

        ############## Save the video and audio clips 
        if method_id == -1: # skip the clip of videos and audios
            continue

       
        if method_id == 12:
            subprocess.run([
                "ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", str(end_time - start_time),
                "-c:v", "libx264", "-preset", "medium", "-an", output_video_filename
            ])

            subprocess.run([ 
                "ffmpeg", "-ss", str(start_time), "-i", audio_path, "-t", str(end_time - start_time),
                "-c:a", "aac", output_audio_filename
            ])
        ##############################################

       
        # ######################################### 
        if method_id == 'frame':

            
            video_command = [
                'ffmpeg',
                '-i', video_path,  
                '-vf', f"select='between(n\,{start_frame}\,{end_frame-1})'", 
                '-r', f'{fps}/1',  
                '-an',  
                output_video_filename  
            ]
            
            audio_command = [
                'ffmpeg', '-ss', str(start_time), '-i', audio_path, '-t', str(duration),
                '-c:a', 'aac', output_audio_filename  
            ]

            subprocess.run(video_command, check=True)
            subprocess.run(audio_command, check=True)
        ##############################################

        # ######################################### 
        if method_id == 'frame_new':

            cut_video_segment(video_path, start_frame, end_frame - 1, output_video_filename)
            
            audio_command = [
                'ffmpeg', '-ss', str(start_time), '-i', audio_path, '-t', str(duration),
                '-c:a', 'aac', output_audio_filename  
            ]

            
            subprocess.run(audio_command, check=True)
        ##############################################

        

        # Now let's save the first frame of the video scene
        scene_video_cap = cv2.VideoCapture(output_video_filename)
        success, first_frame = scene_video_cap.read()
        if success:
            first_frame_path = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}_first_frame.png")
            cv2.imwrite(first_frame_path, first_frame)  # Save the first frame image
            save_flag = True
        else:
            os.remove(output_video_filename)
            os.remove(output_audio_filename)
            save_flag = False
        scene_video_cap.release()

        video_cap_output = cv2.VideoCapture(output_video_filename, cv2.CAP_FFMPEG) 

        if not video_cap_output.isOpened():
            print(f"Error: Could not open video file {output_video_filename}")
            return

        width = int(video_cap_output.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap_output.get(cv2.CAP_PROP_FRAME_HEIGHT))
        son_fps = video_cap_output.get(cv2.CAP_PROP_FPS)
        son_frame_count = video_cap_output.get(cv2.CAP_PROP_FRAME_COUNT)  
        video_cap_output.release()

        start_time_in_original_video = start_time + item['start-time']
        end_time_in_original_video = end_time + item['start-time']

        print(f"subscene {scene_counter} is ready to save infos.")

        if save_flag:
            son_item = item
            son_item.update({
                "id": f"video{video_id}-scene{scene_counter}",
                "video-path": output_video_filename,
                "audio-path": output_audio_filename,
                "first-frame-path": first_frame_path,  # Add the first frame path to the scene data
                "height": height,
                "width": width,
                "fps": son_fps,
                "start-time": start_time_in_original_video,
                # "start-frame": scene[0].get_frames(),

                "end-time": end_time_in_original_video,
                # "end-frame": scene[1].get_frames(),
                "durations": f"{round(son_frame_count/son_fps, 3)}s",
                "father-video": video_path,
                "father-audio": audio_path,
                "video_url": video_url,
                "subtitle_path": subtitle_path,
                "frame_count": son_frame_count,

                "start_time_in_father_video": start_time,
                "end-time_in_father_video": end_time,
            
            })
            scenes_data.append(son_item.copy())
        
        scene_counter += 1  # Increment scene counter

    return scenes_data, start_end_time_of_scenes, True



def find_scenes_clip(video_path, audio_path, output_subfolder, video_id, 
                    video_url, subtitle_path, use_fixed_duration, 
                    detector_type, detector_parameter):
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)


    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_cap.release()

    # Use scenedetect for scene detection
    video = open_video(video_path)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    ### Choose different detectors
    
    threshold = float(detector_parameter)
    if detector_type == "Adaptive":
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold)) 

    elif detector_type == "Histogram":
        scene_manager.add_detector(HistogramDetector(threshold = threshold))
    
    elif detector_type == "Content":
        scene_manager.add_detector(ContentDetector(threshold = threshold))
    
    elif detector_type == "Hash":
        scene_manager.add_detector(HashDetector(threshold = threshold))
    
    elif detector_type == "Threshold":
        scene_manager.add_detector(ThresholdDetector(threshold = threshold))

    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    scenes_data = []
    scene_counter = 1  # Initialize scene counter

    
    save_path_scenes_info = os.path.join(output_subfolder, f"0000_video{video_id}.txt")
     #
    start_end_time_of_scenes = ""

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds() + 0.2 
        end_time = scene[1].get_seconds()
        duration = end_time - start_time

        # Discard segments shorter than 2 seconds
        if duration < 2.0:
            continue  

        if use_fixed_duration:
            # Adjust segments to match the desired durations
            if 5.0 <= duration < 10.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 2.5
                end_time = middle + 2.5
            
            elif 10.0 <= duration < 15.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 5.0
                end_time = middle + 5.0
                
            elif 15.0 <= duration < 20.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 7.5
                end_time = middle + 7.5
            elif 20.0 <= duration < 25.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 10.0
                end_time = middle + 10.0
                
            elif duration >= 25.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 12.5
                end_time = middle + 12.5
        else:
            # Adjust start and end times by subtracting 0.1 seconds
            start_time = max(0, start_time - 0.1)
            end_time = max(0, end_time - 0.1)   

        output_video_filename = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}.mp4")
        output_audio_filename = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}.m4a")

        ## Debugging
        start_minutes, start_seconds = divmod(start_time, 60)
        end_minutes, end_seconds = divmod(end_time, 60)
        

        scene_info = (
            "scene " + str(scene_counter) +
            " infos: start_time " + str(int(start_minutes)) + ":" + str(round(start_seconds, 2)) +
            ", end_time " + str(int(end_minutes)) + ":" + str(round(end_seconds, 2)) + "\n"
        )

        start_end_time_of_scenes += scene_info
        
        os.makedirs(output_subfolder, exist_ok=True)

        
        with open(save_path_scenes_info, "a") as file:  
            file.write(f"scene {scene_counter} infos: start_time {int(start_minutes)}:{int(start_seconds)}, end_time {int(end_minutes)}:{int(end_seconds)}\n")

        
        subprocess.run([
                "ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", str(end_time - start_time),
                "-c:v", "libx264", "-preset", "medium", "-an", output_video_filename
            ])

        subprocess.run([ 
                "ffmpeg", "-ss", str(start_time), "-i", audio_path, "-t", str(end_time - start_time),
                "-c:a", "aac", output_audio_filename
            ])
        ##############################################

        

        # Now let's save the first frame of the video scene
        scene_video_cap = cv2.VideoCapture(output_video_filename)
        success, first_frame = scene_video_cap.read()
        if success:
            first_frame_path = os.path.join(output_subfolder, f"video{video_id}_scene{scene_counter}_first_frame.png")
            cv2.imwrite(first_frame_path, first_frame)  # Save the first frame image
            save_flag = True
        else:
            os.remove(output_video_filename)
            os.remove(output_audio_filename)
            save_flag = False
        scene_video_cap.release()

        

        if save_flag:
            scenes_data.append({
                "id": f"video{video_id}-scene{scene_counter}",
                "video-path": output_video_filename,
                "audio-path": output_audio_filename,
                "first-frame-path": first_frame_path,  # Add the first frame path to the scene data
                "height": height,
                "width": width,
                "fps": fps,
                "start-time": start_time,
                "start-frame": scene[0].get_frames(),

                "end-time": end_time,
                "end-frame": scene[1].get_frames(),
                "durations": f"{round(end_time - start_time, 1)}s",
                "original-video": video_path,
                "original-audio": audio_path,
                "video_url": video_url,
                "subtitle_path": subtitle_path,
            
            })
        
        scene_counter += 1  # Increment scene counter

    return scenes_data, start_end_time_of_scenes, True







def extract_frames_uniform(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < num_frames:
        raise ValueError(f"Video must have at least {num_frames} frames for uniform sampling.")

    frames_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
    print("frames_indices",frames_indices)
    extracted_frames = []
    extracted_frames_path = []
    extracted_frame_indices = []
    for idx, frame_idx in enumerate(frames_indices):
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            extracted_frames.append(img)
            frame_filename = f"{os.path.basename(video_path.split('/')[-1][:-4])}_frame_{frame_idx}.png"
            frame_path = os.path.join(os.path.dirname(video_path), frame_filename)
            print("frame_path",frame_path)
            img.save(frame_path)
            # assert False
            extracted_frames_path.append(frame_path)
            extracted_frame_indices.append(frame_idx)
        else:
            print(f"Failed to extract frame at index {frame_idx} from {video_path}")
    cap.release()
    return extracted_frames, extracted_frames_path, extracted_frame_indices


def predict_scene_batch(model, centre_crop, classes, frames, device):
    input_imgs = torch.stack([centre_crop(frame) for frame in frames]).to(device)
    input_imgs = V(input_imgs)
    logits = model(input_imgs)
    h_x = F.softmax(logits, dim=1).data
    probs, idx = h_x.sort(1, True)
    predictions = [[classes[idx[i][j]] for j in range(5)] for i in range(len(frames))]
    return predictions


def process_video(model, centre_crop, classes, video_data, device):
    video_path = video_data['video-path']
    try:
        duration_str = video_data['durations']
        duration_seconds = float(duration_str[:-1]) if duration_str.endswith('s') else 0
        num_frames = max(5, math.floor(duration_seconds))  # Ensure at least 5 frames or rounded duration
        extracted_frames,extracted_frames_path,extracted_frame_indices = extract_frames_uniform(video_path, num_frames)

        if not extracted_frames:
            raise ValueError("No frames were extracted from the video.")

        all_predictions = []
        try:
            predictions = predict_scene_batch(model, centre_crop, classes, extracted_frames, device)
            for frame_predictions in predictions:
                all_predictions.extend(frame_predictions)
        except Exception as e:
            print(f"Error predicting scenes for frames in {video_path}: {e}")

        if all_predictions:
            scene_counter = Counter(all_predictions)
            most_common_scene = scene_counter.most_common(1)[0][0]
            video_data['scene category'] = most_common_scene
            video_data['keyframes path'] = extracted_frames_path
            video_data['keyframes idx'] = extracted_frame_indices
        else:
            print(f"No predictions were made for {video_path} due to lack of frames.")
            video_data['keyframes path'] = None
            video_data['keyframes idx'] = None

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        video_data['keyframes'] = None

    return video_data


def extract_frames(
    video_path,
    frame_inds=None,
    points=None,
    backend="opencv",
    return_length=False,
    num_frames=None,
):
    """
    Args:
        video_path (str): path to video
        frame_inds (List[int]): indices of frames to extract
        points (List[float]): values within [0, 1); multiply #frames to get frame indices
    Return:
        List[PIL.Image]
    """
    assert backend in ["av", "opencv", "decord"]
    assert (frame_inds is None) or (points is None)

    if backend == "av":
        import av

        container = av.open(video_path)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = container.streams.video[0].frames

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frames = []
        for idx in frame_inds:
            if idx >= total_frames:
                idx = total_frames - 1
            target_timestamp = int(idx * av.time_base / container.streams.video[0].average_rate)
            container.seek(target_timestamp)
            frame = next(container.decode(video=0)).to_image()
            frames.append(frame)

        if return_length:
            return frames, total_frames
        return frames

    elif backend == "decord":
        import decord

        container = decord.VideoReader(video_path, num_threads=1)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = len(container)

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frame_inds = np.array(frame_inds).astype(np.int32)
        frame_inds[frame_inds >= total_frames] = total_frames - 1
        frames = container.get_batch(frame_inds).asnumpy()  # [N, H, W, C]
        frames = [Image.fromarray(x) for x in frames]

        if return_length:
            return frames, total_frames
        return frames

    elif backend == "opencv":
        cap = cv2.VideoCapture(video_path)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frames = []
        for idx in frame_inds:
            if idx >= total_frames:
                idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_MSEC, idx)

            # HACK: sometimes OpenCV fails to read frames, return a black frame instead
            try:
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            except Exception as e:
                print(f"[Warning] Error reading frame {idx} from {video_path}: {e}")
                # First, try to read the first frame
                try:
                    print(f"[Warning] Try reading first frame.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                # If that fails, return a black frame
                except Exception as e:
                    print(f"[Warning] Error in reading first frame from {video_path}: {e}")
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame = Image.new("RGB", (width, height), (0, 0, 0))

            # HACK: if height or width is 0, return a black frame instead
            if frame.height == 0 or frame.width == 0:
                height = width = 256
                frame = Image.new("RGB", (width, height), (0, 0, 0))

            frames.append(frame)

        if return_length:
            return frames, total_frames
        return frames
    else:
        raise ValueError


def convert_json_line_to_general(input_json):
    try: 
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pass
    except json.JSONDecodeError as e:
        try:
            with open(input_json, 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    try:
                        sample = json.loads(line)
                        data.append(sample)
                    except json.JSONDecodeError:
                        # If the line is malformed, attempt to extract a valid JSON fragment
                        sample = extract_valid_json_from_malformed_line(line)
                        if sample and 'original-video' in sample:
                            data.append(sample) 
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")

            with open(input_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4, default=default_dump)
        except json.JSONDecodeError as e:
            print("this is a complete json file")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            exit()


def load_json_data(json_folder, front_prefix, part_number, prefix=None):
    input_json_path = os.path.join(json_folder, f"{front_prefix}_part_{part_number}.json")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # complete or only one
    except json.JSONDecodeError as e:
        data = []
        for line in f:
            sample = json.loads(line)
            data.append(sample)
    existing_ids = set()
    if prefix is None:
        return data, existing_ids
    else:
        output_json_path = os.path.join(json_folder, f"{prefix}_part_{part_number}.json")
        # existing_ids = load_existing_ids(output_json_path)
        try:
            if os.path.isfile(output_json_path):
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    out_data = json.load(f)
                if isinstance(out_data, dict):
                    existing_ids = check_output_file(output_json_path)
                elif isinstance(out_data, list):
                    print(f"part {part_number} completed")
                    for sample in tqdm(out_data, desc="check samples", unit="sample"):
                        existing_ids.add(sample['id'])
        except json.JSONDecodeError as e:
            # if the json format is constructed but not processed completed?
            existing_ids = check_output_file(output_json_path)
            
        return data, existing_ids


def get_existing_video_ids(max_ckpt_file, part_excel_folder=None, json_folder=None, filtered_data_metric=None, front_filtered_data_metric=None):
    '''
    get a set of all existing video IDs from part Excel files
    '''
    existing_ids = set()
    part_number = find_max_number(json_folder, max_ckpt_file, filtered_data_metric)  # Find the next part number
    if filtered_data_metric == 'undetected' or filtered_data_metric is None:
        if filtered_data_metric == 'undetected':
            prefix = 'all_scenes_data_undetected'
        elif filtered_data_metric is None:
            prefix = 'all_scenes_data'
        if part_number == 0:
            part_number = 1
            part_excel_path = os.path.join(part_excel_folder, f"part_{part_number}.csv")
            part_meta_df = pd.read_csv(part_excel_path)
            data = part_meta_df.loc[:, 'video_id'].to_list()
            print(f'processing the first csv file')
            save_max_file(max_ckpt_file, part_number)
            return data, part_number, part_meta_df
        part_excel_path = os.path.join(part_excel_folder, f"part_{part_number}.csv")
        part_meta_df = pd.read_csv(part_excel_path)
        try:
            with open(os.path.join(json_folder, f"{prefix}_part_{part_number}.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)    
            if isinstance(data, dict):
                existing_ids = check_clips_output_file(os.path.join(json_folder, f"{prefix}_part_{part_number}.json"))
            elif isinstance(data, list):
                pattern = r'video(.+?)-'
                for sample in tqdm(data, desc="check samples", unit="sample"):
                    existing_ids.add(extract_video_id(sample['id'], pattern))
        except json.JSONDecodeError as e:
            # if the json format is constructed but not processed completed?
            existing_ids = check_clips_output_file(os.path.join(json_folder, f"{prefix}_part_{part_number}.json"))
        data = part_meta_df.loc[:, 'video_id'].to_list()
        if existing_ids is not None:
            # data = [sample for sample in data if sample['id'] not in existing_ids]
            data = [sample for sample in data if sample not in existing_ids]
        if len(data) == 0:
            convert_json_line_to_general(os.path.join(json_folder, f"{prefix}_part_{part_number}.json")) # Prevents the last process from breaking before covert
            part_number = part_number + 1
            part_excel_path = os.path.join(part_excel_folder, f"part_{part_number}.csv")
            existing_ids = set()
            part_meta_df = pd.read_csv(part_excel_path)
            data = part_meta_df.loc[:, 'video_id'].to_list()
            save_max_file(max_ckpt_file, part_number)
        else:
            part_meta_df = part_meta_df[part_meta_df['video_id'].isin(data)]
            part_meta_df = part_meta_df.reset_index(drop=True)
        print(f'processing part {part_number} csv file')
        return data, part_number, part_meta_df
    elif filtered_data_metric is not None and part_excel_folder is None:
        prefix = f'all_scenes_data_{filtered_data_metric}'
        if front_filtered_data_metric is None:
            front_prefix = f'all_scenes_data'
        else:
            front_prefix = f'all_scenes_data_{front_filtered_data_metric}'
        if part_number == 0:
            part_number = 1
            # Load input IDs
            data, existing_ids = load_json_data(json_folder, front_prefix, part_number)

            print(f'processing the first json file')
            save_max_file(max_ckpt_file, part_number)
            return data, existing_ids, part_number
        
        data, existing_ids = load_json_data(json_folder, front_prefix, part_number, prefix)

        if len([entry for entry in data if entry["id"] not in existing_ids]) == 0:
            convert_json_line_to_general(os.path.join(json_folder, f"{prefix}_part_{part_number}.json")) # Prevents the last process from breaking before covert
            part_number = part_number + 1
            data, existing_ids = load_json_data(json_folder, front_prefix, part_number, prefix)
            save_max_file(max_ckpt_file, part_number)
        print(f'processing part {part_number} csv file')
        return data, existing_ids, part_number
    

    ###############################################

    def calculate_total_video_time(json_path, detector_type, detector_parameter):
        import matplotlib.pyplot as plt
        import numpy as np
        with open(json_path, 'r') as f:
            data = json.load(f)

        durations = []

        for item in data:
            duration_str = item["durations"]
            duration = float(duration_str.replace("s", ""))
            durations.append(duration)

        max_duration = max(durations)
        bins = np.arange(0, max_duration + 5, 5)  # 5 seconds for each bin

        plt.figure(figsize=(12, 6))
        plt.hist(durations, bins=bins, edgecolor='black', alpha=0.7)

        # y axis in log scale
        plt.yscale('log')

        plt.xlim(0, max(durations) * 1.1)

        plt.xticks(np.arange(0, max(durations), step=50))

        plt.title("Video Duration Distribution")
        plt.xlabel("Duration (s)")
        plt.ylabel("Frequency (log scale)")

        plt.grid(True)

        folder_path = os.path.dirname(json_path)
        plot_path = os.path.join(folder_path, f"{detector_type}_{detector_parameter}_video_duration_distribution.png")
        plt.savefig(plot_path)
        plt.close() 

        print(f"save the figure to {plot_path}")

    def duration_check(json_path, duration_threshold):
        with open(json_path, 'r') as f:
            data = json.load(f)
        video_cliped_checked = []

        for item in data:
            duration_str = item["durations"]
            duration = float(duration_str.replace("s", ""))
            if duration > duration_threshold:
                video_cliped_checked.append(item)
        new_json_path = json_path.replace(".json", f"_duration_threshold={duration_threshold}.json")
        save_json_entry(video_cliped_checked, new_json_path)

        convert_json_line_to_general(json_path)
        return new_json_path
     ###############################################