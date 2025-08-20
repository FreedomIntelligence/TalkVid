#!/usr/bin/env python3
# multi_gpu_runner.py

import os
import json
import yaml
import math
import shutil
import subprocess
import uuid
import time 
import logging
from datetime import datetime
import multiprocessing as mp
from functools import partial

def setup_logger(config):
    """
    setup_logger(config) -> log_file, timestamp
    """
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', 'cotracker', timestamp+f"_{os.path.basename(config['json_path']).replace('.json', '')}")
    os.makedirs(log_dir, exist_ok=True)
    
    # create log file path
    log_file = os.path.join(log_dir, f'cotracker_main.log')
    
    # remove existing handlers to avoid duplicate logs
    logging.getLogger().handlers = []
    
    # basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # print to console as well
        ]
    )
    
    # logging configuration
    logging.info("Starting cotracker processing with configuration:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    
    return log_file, timestamp

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)['cotracker']
    return config



def split_json_into_chunks(json_path, num_chunks, tmp_dir="tmp_splits"):
    """
    Divide the JSON file into num_chunks parts based on the duration of each item.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_ids = set()
    
    processed_data_files = os.listdir(tmp_dir)
    processed_data_files = [file for file in processed_data_files if 'out.json' in file]
    for file in processed_data_files:
        try:
            with open(os.path.join(tmp_dir, file), "r", encoding="utf-8") as f:
                processed_data = []
                for line in f:
                    if line.strip():
                        processed_data.append(json.loads(line))
                        processed_ids.update(item["id"] for item in processed_data if item['cotracker_ratio'] is not None)
        except (json.JSONDecodeError, FileNotFoundError):
            logging.warning(f"Warning: {file} fails to load or is empty.")
            os.remove(os.path.join(tmp_dir, file))

    processed_ids = set()
    
    unprocessed_data = [item for item in data if item["id"] not in processed_ids]
    logging.info(f"the item has been dealed: {len(processed_ids)}")
    logging.info(f"the item hasn't been dealed: {len(unprocessed_data)}")
    
    if not unprocessed_data:
        return False

    # sort unprocessed_data by duration
    for item in unprocessed_data:
        item['duration_float'] = float(item["durations"].replace("s", ""))
    sorted_data = sorted(unprocessed_data, key=lambda x: x['duration_float'], reverse=True)


    # use greedy algorithm to distribute items into chunks
    chunks = [[] for _ in range(num_chunks)]
    chunk_durations = [0.0] * num_chunks

    
    for item in sorted_data:
        min_duration_idx = chunk_durations.index(min(chunk_durations))
        chunks[min_duration_idx].append(item)
        chunk_durations[min_duration_idx] += item['duration_float']
        
        item.pop('duration_float', None)

    
    chunk_paths = []
    for i in range(num_chunks):
        chunk_path = os.path.join(tmp_dir, f"chunk_{i}.json")
        with open(chunk_path, "w", encoding="utf-8") as cf:
            json.dump(chunks[i], cf, ensure_ascii=False, indent=4)
        chunk_paths.append(chunk_path)
        logging.info(f"Chunk {i} total duration: {chunk_durations[i]:.2f}s")

    return chunk_paths

    

def worker(chunk_path, config, script_path, device_id, timestamp):
    """
    worker function for processing each chunk
    """
    
    log_dir = os.path.join('logs', 'cotracker', timestamp)
    os.makedirs(log_dir, exist_ok=True)
    worker_log_file = os.path.join(log_dir, f'worker_gpu{device_id}.log')
    
    
    logging.getLogger().handlers = []
    
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - GPU{device_id} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(worker_log_file, encoding='utf-8'),
            logging.StreamHandler()  
        ]
    )

    import sys
    import importlib.util

    
    spec = importlib.util.spec_from_file_location("camera_filtering", script_path)
    camera_filtering = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera_filtering)

    
    class Args:
        pass
    
    args = Args()
    args.json_path = chunk_path
    args.output_json = chunk_path.replace(".json", "_out.json")
    args.checkpoint = config["checkpoint"]
    args.grid_size = config.get("grid_size", 50)
    args.grid_query_frame = config.get("grid_query_frame", 0)
    args.backward_tracking = config["backward_tracking"]
    args.offline = config["offline"]
    args.device_id = device_id
    args.gpu_preprocess = config.get("gpu_preprocess", False)
    args.timestamp = timestamp
    args.use_v2_model = False

    try:
        
        camera_filtering.main(args)
        return True, args.output_json
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_path}: {str(e)}")
        return False, None

def run_subprocesses_on_gpus(chunks, config, script_path="camera_filtering.py", timestamp=None):
    """
    multiprocessing chunks
    """
    num_gpus = config["num_gpus"]
    runs_per_gpu = config["runs_per_gpu"]
    device_ids = config["device_ids"]
    max_concurrent_processes = num_gpus * runs_per_gpu

    
    pool = mp.Pool(processes=max_concurrent_processes)

    
    tasks = []
    for idx, chunk_path in enumerate(chunks):
        gpu_index = idx // runs_per_gpu
        device_id = device_ids[gpu_index % len(device_ids)]
        
        
        task = (chunk_path, config, script_path, device_id, timestamp)
        tasks.append(task)

    
    worker_with_args = partial(worker)
    
    
    results = []
    for task in tasks:
        result = pool.apply_async(worker_with_args, task)
        results.append((task[0], result))  

   
    pool.close()

    
    processes = []
    for chunk_path, result in results:
        try:
            success, output_json = result.get()  
            if success:
                logging.info(f"Successfully processed chunk: {chunk_path}")
                processes.append((len(processes), None, chunk_path, output_json))
            else:
                logging.error(f"Failed to process chunk: {chunk_path}")
        except Exception as e:
            logging.error(f"Error getting result for chunk {chunk_path}: {str(e)}")

    
    pool.join()

    return processes

def wait_and_check(processes):
    """
    If it is a result object from multiprocess.Pool.apply_async(), there is no need to call p.communicate(). 
    You can directly determine whether the execution was successful based on the worker's return value.
    """
    output_files = []

    try:
        for (idx, p, chunk_path, out_json) in processes:
            if out_json:
                output_files.append(out_json)
    except Exception as e:
        logging.error(f"Error checking result: {e}")

    return output_files

def merge_json_files(output_files, final_output):
    """
    Merge all output_files (sub-results) into a single final_output file.
    Assume that the structure of all sub-results is a list of dictionaries.
    If you need to deduplicate by ID or implement merging logic, more complex processing can be implemented here.
    """
    merged_data = []
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    try:
        for ofile in output_files:
            if not os.path.exists(ofile):
                continue
            with open(ofile, "r", encoding="utf-8") as f:
                part = []
                for line in f:
                    try:
                        d = json.loads(line)
                        if d['cotracker_ratio'] is not None:
                            part.append(d)
                    except Exception as e:
                        logging.error(f"Error loading line: {line}")
                # part = json.load(f)
                merged_data.extend(part)
    except Exception as e:
        logging.error("Error merging results!")

    # If deduplication based on "id" is needed, additional processing can be done here.
    # e.g., use a dictionary to store id -> item

    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--json_path", type=str, help="Path to the input JSON file, overrides the config file")
    parser.add_argument("--output_json", type=str, help="Path to the output JSON file, overrides the config file")
    parser.add_argument("--device_ids", type=str, help="List of GPU IDs to use, separated by commas, e.g., '0,1,2,3'")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs to use, overrides the config file")
    parser.add_argument("--runs_per_gpu", type=int, help="Number of processes to run on each GPU, overrides the config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Command-line arguments override settings in the configuration file
    if args.json_path:
        config["json_path"] = args.json_path
    if args.output_json:
        config["output_json"] = args.output_json
    if args.device_ids:
        config["device_ids"] = [int(x) for x in args.device_ids.split(',')]
    if args.num_gpus:
        config["num_gpus"] = args.num_gpus
    if args.runs_per_gpu:
        config["runs_per_gpu"] = args.runs_per_gpu
    
    # logging setup
    log_file, timestamp = setup_logger(config)
    logging.info(f"Log file created at: {log_file}")
    logging.info(f"Using GPU IDs: {config['device_ids']}")
    logging.info(f"GPU num: {config['num_gpus']}")
    logging.info(f"Process per GPU: {config['runs_per_gpu']}")

    # 1. Divide the JSON file into chunks
    total_subtasks = config["num_gpus"] * config["runs_per_gpu"]
    if config["gpu_preprocess"]:
        tmp_dir = os.path.join('temp_dir_gpu_new', os.path.basename(config["json_path"]).replace(".json", "_tmp"))
    else:
        tmp_dir = os.path.join('temp_dir_new', os.path.basename(config["json_path"]).replace(".json", "_tmp"))
    os.makedirs(tmp_dir, exist_ok=True)
    chunks = split_json_into_chunks(config["json_path"], total_subtasks, tmp_dir=tmp_dir)

    # 2. Distribute the chunks to different GPUs
    if chunks:
        processes = run_subprocesses_on_gpus(chunks, config, script_path="camera_filtering.py", timestamp=timestamp)

        # 3. Wait for all processes to finish and collect output files
        output_files = wait_and_check(processes)
    else:
        logging.info("All samples have been processed.")
        output_files = []
        output_files = os.listdir(tmp_dir)
        output_files = [os.path.join(tmp_dir, file) for file in output_files if file.endswith("_out.json")]

    # 4. Merge all output files into a single final output file
    merge_json_files(output_files, config["output_json"])

    logging.info(f"All done. Final merged output => {config['output_json']}")
    
    # Total processing time
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    main()

'''
python multi_gpu_runner.py --config config.yaml
'''