import os
import json
import torch
import argparse
from tqdm import tqdm
from cotracker.predictor import CoTrackerPredictor
import numpy as np
import cv2  
import time
import math
import decord
from decord import VideoReader
import logging
from datetime import datetime
decord.bridge.set_bridge('torch')

# Set DECORD_EOF_RETRY_MAX 
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

import torchvision.transforms as T
import numpy as np

def get_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    free_memory = total_memory - reserved_memory
    return allocated_memory, reserved_memory, free_memory

def find_max_batch_size(frames_batch, new_width, new_height, device='cuda'):
    """
    Find maximum batch size for GPU
    """
    batch_size = 1
    while True:
        try:
            # Try to load all data for the current batch
            batch_tensor = torch.from_numpy(frames_batch[:batch_size]).to(device)
            resize_transform = T.Resize((new_height, new_width), 
                                        interpolation=T.InterpolationMode.BILINEAR)
            
            # resize
            with torch.no_grad():
                resized_frames = resize_transform(batch_tensor.permute(0, 3, 1, 2))
                resized_frames = resized_frames.permute(0, 2, 3, 1)
            
            # If successful, increase the batch size
            batch_size += 1
        except RuntimeError as e:
            # If out of memory, raise an error and return the last successful batch size
            print(f"Error: {e}, batch_size: {batch_size}")
            break
    
    # return the last successful batch size
    return max(1, batch_size - 5)

def torch_gpu_resize_batch(frames_batch, new_width, new_height, device='cuda'):
    """
    adjust batch size based on GPU memory
    """
    # Get current GPU memory usage
    allocated_memory, reserved_memory, free_memory = get_gpu_memory()
    # print(f"Free GPU memory: {free_memory / 1024**2:.2f} MB")

    # Dynamically find the maximum batch size
    # batch_size = find_max_batch_size(frames_batch, new_width, new_height, device)
    batch_size = 80

    # logging.info(f"Max batch size: {batch_size}")
    
    resized_frames = []
    # Resize using the maximum batch size found
    for i in range(0, len(frames_batch), batch_size):
        frames_tensor = torch.from_numpy(frames_batch[i:i+batch_size]).to(device)
        resize_transform = T.Resize((new_height, new_width), 
                                    interpolation=T.InterpolationMode.BILINEAR)
        
        with torch.no_grad():
            resized_frame = resize_transform(frames_tensor.permute(0, 3, 1, 2))
            
            resized_frame = resized_frame.permute(0, 2, 3, 1)
        
        resized_frames.append(resized_frame)

    resized_frames = torch.cat(resized_frames, dim=0)
    return resized_frames.cpu().numpy()


def resize_batch(frames_batch, new_width, new_height):
    resized_batch = np.stack([cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA) 
                            for frame in frames_batch])
    return resized_batch

def setup_logger(args):
    """
    Set up the logger to log 
    """
    # # Use the same directory if a timestamp is provided, otherwise create a new one.
    if args.timestamp:
        log_dir = os.path.join('logs', 'cotracker', args.timestamp)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join('logs', 'cotracker', timestamp)
    
    os.makedirs(log_dir, exist_ok=True)
    
    # create a log file with the device ID
    log_file = os.path.join(log_dir, f'cotracker_camera_filtering_gpu{args.device_id}.log')
    
    # logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # print to console as well
        ]
    )
    
    # Log the parameters
    logging.info("Starting camera filtering with arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    
    return log_file


def read_video_from_path(path, desired_fps=30, scale_factor=0.5, device_id=0, gpu_preprocess=False):
    # use decord to read video
    start_time = time.perf_counter()
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            vr = VideoReader(
                path,
                decord.cpu(0),
                num_threads= 4  # increase number of threads for faster reading
            )
            break
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                logging.error(f"Error reading video {path} after {max_retries} attempts: {str(e)}")
                return None, None, None
            logging.warning(f"Retry {retry_count}/{max_retries} reading video {path}: {str(e)}")
            time.sleep(1)  # wait before retrying
            
    read_time = time.perf_counter() - start_time
    
    original_fps = vr.get_avg_fps()
    
    if original_fps <= 0:
        logging.warning(f"Unable to retrieve FPS for video: {path}. Defaulting to desired FPS: {desired_fps}.")
        original_fps = desired_fps

    # calculate the frame interval based on the desired FPS
    if original_fps > desired_fps:
        frame_interval = original_fps / desired_fps
        # calculate the frame indices to sample
        total_frames = len(vr)
        frame_indices = [int(i * frame_interval) for i in range(int(total_frames / frame_interval))]
    else:
        frame_indices = list(range(len(vr)))

    # 1000 frames per batch
    batch_size = 1000
    processed_frames = []

    pre_process_start = time.perf_counter()
    load_batch_time = 0.0
    reshape_batch_time = 0.0
    resize_batch_time = 0.0

    for i in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[i:i + batch_size]
        load_batch_start = time.perf_counter()
        # read current batch of frames
        try:
            frames_batch = vr.get_batch(batch_indices).numpy()
        except Exception as e:
            logging.error(f"Error reading video from indices {path}: {str(e)}")
            return None, None, None
        load_batch_end = time.perf_counter()
        load_batch_time += load_batch_end - load_batch_start
        # Batch process the resize operation
        try:
            height, width = frames_batch[0].shape[:2]
            new_width = max(1, int(width * scale_factor))
            new_height = max(1, int(height * scale_factor))
            # Use NumPy's vectorized operations instead of a loop
            reshape_batch_start = time.perf_counter()
            frames_reshaped = frames_batch.reshape(-1, height, width, 3)
            reshape_batch_end = time.perf_counter()
            reshape_batch_time += reshape_batch_end - reshape_batch_start
            resize_batch_start = time.perf_counter()
            if gpu_preprocess:
                resized_batch = torch_gpu_resize_batch(frames_reshaped, new_width, new_height, device=f'cuda:{device_id}')
            else:
                resized_batch = resize_batch(frames_reshaped, new_width, new_height)
            resize_batch_end = time.perf_counter()
            resize_batch_time += resize_batch_end - resize_batch_start
            processed_frames.extend(resized_batch)

        except Exception as e:
            logging.error(f"Error resizing frames: {str(e)}")
            return None, None, None
            

    pre_process_time = time.perf_counter() - pre_process_start

    if not processed_frames:
        logging.error(f"No frames read from video: {path}")
        return None, None, None

    # remove first 0.2 seconds and last 0.2 seconds
    fps = desired_fps if original_fps > desired_fps else original_fps
    if len(processed_frames) > fps:
        processed_frames = processed_frames[int(0.2 * fps):-int(0.2 * fps)]
    
    # Convert the processed frames to a NumPy array
    try:
        stack_start = time.perf_counter()
        result = np.stack(processed_frames)
        stack_end = time.perf_counter()
        stack_time = stack_end - stack_start
    except Exception as e:
        logging.error(f"Error stacking frames: {str(e)}")
        return None, None, None
    
    # Clean up the original video reader
    del vr
    total_pre_process_time = time.perf_counter() - start_time

    return result, fps, (read_time, pre_process_time, total_pre_process_time, load_batch_time, reshape_batch_time, resize_batch_time, stack_time)



def parse_duration(duration_str):
    """
    convert duration string to float.
    e.g. "74.1s" -> 74.1
    """
    try:
        return float(duration_str.strip().rstrip('s'))
    except ValueError:
        logging.warning(f"Unable to parse duration '{duration_str}'. Defaulting to 0.")
        return 0.0


def get_segments(duration):
    """
    Returns a list of segments to be processed based on the duration.
    Each segment consists of (start_time, end_time) in seconds.
    """
    segments = []
    if duration <= 6.0:
        segments.append((0.0, duration))
    elif 6.0 < duration <= 12.0:
        segments.append((0.0, 6.0))
        segments.append((duration - 6.0, duration))
    else:
        segments.append((0.0, 6.0))
        middle_start = (duration / 2) - 3.0
        segments.append((middle_start, middle_start + 6.0))
        segments.append((duration - 6.0, duration))
    return segments


def extract_frames(video_np, fps, sample_fps, start_time, end_time):
    """
    Extracts the corresponding frames based on the start and end times.
    """
    start_frame = int(math.floor(start_time * fps))
    end_frame = int(math.ceil(end_time * fps))
    
    start_frame = max(start_frame, 0)
    end_frame = min(end_frame, video_np.shape[0])
    # sample frames at the specified sample_fps
    sample_interval = int(fps / sample_fps)
    sampled_indices = list(range(start_frame, end_frame, sample_interval))
    sampled_frames = video_np[sampled_indices]
    return sampled_frames


def main(args=None):
    """
    Modify the main function to accept arguments
    """
    if args is None:
        parser = argparse.ArgumentParser(
            description="Batch script for CoTracker ratio calculation on multiple videos with incremental saving."
        )
        parser.add_argument(
            "--json_path",
            required=True,
            help="Path to the input JSON file that lists multiple video items",
        )
        parser.add_argument(
            "--output_json",
            default="output.json",
            help="Path to the output JSON where we save partial or final results",
        )
        parser.add_argument(
            "--checkpoint",
            default="./checkpoints/scaled_offline.pth",
            help="Path to the CoTracker model checkpoint",
        )
        parser.add_argument("--grid_size", type=int, default=50, help="Regular grid size")
        parser.add_argument(
            "--grid_query_frame",
            type=int,
            default=0,
            help="Compute dense and grid tracks starting from this frame",
        )
        parser.add_argument(
            "--backward_tracking",
            action="store_true",
            help="Compute tracks in both directions, not only forward",
        )
        parser.add_argument(
            "--use_v2_model",
            action="store_true",
            help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Pass it if you would like to use the offline model, in case of online don't pass it",
        )
        parser.add_argument(
            "--device_id",
            type=int,
            default=0,
            help="Which CUDA device to use, e.g. 0 or 1. If no GPU is available, fallback to CPU.",
        )
        parser.add_argument(
            "--gpu_preprocess",
            action="store_true",
            help="Use GPU for preprocessing if set",
        )
        parser.add_argument(
            "--timestamp",
            type=str,
            default="",
            help="Timestamp for log directory organization",
        )
        args = parser.parse_args()

    # Set up logging
    log_file = setup_logger(args)
    logging.info(f"Log file created at: {log_file}")

    if torch.cuda.is_available():
        device = f"cuda:{args.device_id}"
    else:
        device = "cpu"
    logging.info(f"Using device: {device}")

    with open(args.json_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    existing_results_dict = {}
    if os.path.isfile(args.output_json):
        # read json by line
        with open(args.output_json, "r", encoding="utf-8") as f:
            try:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if "id" in item:
                            existing_results_dict[item["id"]] = item
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logging.warning(f"Error reading {args.output_json}: {str(e)}")
    
    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=True)
        else:
            if args.offline:
                window_len = 60
            else:
                window_len = 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=False,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    logging.info("Model loaded successfully")

    for item in tqdm(input_data, desc="tracking"):
        try:
            # Add monitoring for GPU memory usage
            if torch.cuda.is_available():
                logging.info(f"GPU {args.device_id} memory before processing: "
                           f"Allocated: {torch.cuda.memory_allocated(args.device_id)/1024**2:.2f}MB, "
                           f"Cached: {torch.cuda.memory_reserved(args.device_id)/1024**2:.2f}MB")
            
            process_start = time.perf_counter()
            
            item_id = item.get("id", None)
            video_path = item.get("video-path", None)

            if not item_id or not video_path:
                continue

            if item_id in existing_results_dict:
                if "cotracker_ratio" in existing_results_dict[item_id]:
                    logging.info(f"[{item_id}] Already processed, skipping.")
                    continue

            duration_str = item.get("durations", "0s")
            duration = parse_duration(duration_str)

            if not os.path.exists(video_path):
                logging.warning(f"Video path not found: {video_path}")
                item["cotracker_ratio"] = None
                item["processing_time_cotracker"] = None
                _save_or_update_item_in_json(item, existing_results_dict, args.output_json)
                continue

            video_np, fps, (read_time, pre_process_time, total_pre_process_time, load_batch_time, reshape_batch_time, resize_batch_time, stack_time) = read_video_from_path(video_path, desired_fps=30, scale_factor=0.25, device_id=args.device_id, gpu_preprocess=args.gpu_preprocess)
            
            if video_np is None:
                logging.error(f"[{item_id}] Failed to read video.")
                item["cotracker_ratio"] = None
                item["processing_time_cotracker"] = None
                _save_or_update_item_in_json(item, existing_results_dict, args.output_json)
                continue

            sample_fps = 2.0
            segments = get_segments(duration)

            min_ratio = math.inf
            total_processing_time = 0.0

            segments_start = time.perf_counter()
            for idx, (start_time, end_time) in enumerate(segments):
                try:
                    segment_start = time.perf_counter()
                    
                    # Extract frames for the current segment
                    segment_frames = extract_frames(video_np, fps, sample_fps, start_time, end_time)
                    if segment_frames.size == 0:
                        logging.warning(f"[{item_id}] Segment {idx+1} has no frames, skipping.")
                        continue

                    # convert to tensor and add batch dimension
                    video_tensor = torch.from_numpy(segment_frames).permute(0, 3, 1, 2)[None].float().to(device)
                    
                    # model inference
                    with torch.no_grad():
                        pred_tracks, pred_visibility = model(
                            video_tensor,
                            grid_size=args.grid_size,
                            grid_query_frame=args.grid_query_frame,
                            backward_tracking=args.backward_tracking,
                        )
                    
                    ratio = pred_visibility.float().mean().item()
                    segment_time = time.perf_counter() - segment_start
                    total_processing_time += segment_time
                    # logging.info(f"[{item_id}] Segment {idx+1} ratio: {ratio:.4f}, processing time: {segment_time:.2f}s")

                    if ratio < min_ratio:
                        min_ratio = ratio

                    # clean up tensors
                    del video_tensor, pred_tracks, pred_visibility
                    torch.cuda.empty_cache()

                except Exception as e:
                    logging.error(f"[{item_id}] Segment {idx+1} processing failed: {e}")
                    continue


            if min_ratio == math.inf:
                min_ratio = None
                total_processing_time = None

            item["cotracker_ratio"] = min_ratio
            item["processing_time_cotracker"] = total_processing_time
            
            # save results
            _save_or_update_item_in_json(item, existing_results_dict, args.output_json)
            
            total_item_time = time.perf_counter() - process_start
            # Record a summary of all timing information
            logging.info(
                f"[{item_id}] Time summary:\n"
                f"  - Video path: {item['video-path']}\n"
                f"  - Video duration: {duration:.2f}s\n"
                f"  - Video loading time: {read_time:.2f}s\n"
                f"  - Load batch time: {load_batch_time:.2f}s\n"
                f"  - Reshape batch time: {reshape_batch_time:.2f}s\n"
                f"  - Resize batch time: {resize_batch_time:.2f}s\n"
                f"  - Stack time: {stack_time:.2f}s\n"
                f"  - Total preprocessing time: {total_pre_process_time:.2f}s\n"
                f"  - Total inferencing time: {total_processing_time:.2f}s\n"
                f"  - Total item processing took: {total_item_time:.2f}s"
            )

            # clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # logging.info(f"GPU {args.device_id} memory after processing: "
                #            f"Allocated: {torch.cuda.memory_allocated(args.device_id)/1024**2:.2f}MB, "
                #            f"Cached: {torch.cuda.memory_reserved(args.device_id)/1024**2:.2f}MB")

        except Exception as e:
            logging.error(f"Error processing item {item_id}: {str(e)}")
            # clear GPU memory if error occurs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    logging.info("All done.")


def _save_or_update_item_in_json(new_item, existing_dict, output_json):
    """
    Append new results to a JSON file.
    """

    try:
        item_id = new_item["id"]
        existing_dict[item_id] = new_item
        
        
        if not os.path.exists(output_json):
            with open(output_json, "w", encoding="utf-8") as f:
                f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            return
        
        
        with open(output_json, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
        
        logging.info(f"Item {item_id} saved to {output_json}")
    except Exception as e:
        logging.error(f"Error saving item {item_id}: {str(e)}")


if __name__ == "__main__":
    main()

