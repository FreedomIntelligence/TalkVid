import subprocess
import time
import multiprocessing
import json
import yaml
import argparse


def run_command(args):
    gpu_id, limit_videos, config_path = args
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python -u src/video_head_filter.py --use_folder_to_save_results=True --limit_videos=\"{limit_videos}\" --config_path={config_path}"
    
    start_time = time.time()
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  
        text=True,
        bufsize=1, 
        encoding='utf-8'
    )
    
    print(f"\n[GPU {gpu_id}, {limit_videos}] Execute:")
    for line in process.stdout:
        print(f"[GPU {gpu_id}, {limit_videos}] {line}", end="")
    
    process.wait()  
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"[GPU {gpu_id}, {limit_videos}] Finished, takes: {execution_time:.2f} s")

def main():
    # config_path = "path/to/your/config.yaml"  # Update this path accordingly
    parser = argparse.ArgumentParser(description="Run video head filter with multiprocessing.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()
    
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)['head_filter']
    
    json_path = config['input_path']
    num_processes_per_gpu = config['num_processes']
    gpu_ids = config['cuda']['devices']
    print(f"gpu_ids = {gpu_ids}")
    print(f"config_path = {config_path}")
    
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_items = len(data)
    num_gpus = len(gpu_ids)
    total_processes = num_gpus * num_processes_per_gpu
    chunk_size = total_items // total_processes + (1 if total_items % total_processes else 0)
    print(f"total_processes = {total_processes}")
    print(f"chunk_size = {chunk_size}")
    
    video_ranges = [(i * chunk_size, min((i + 1) * chunk_size - 1, total_items - 1)) for i in range(total_processes)]
    limits = [f"{start},{end}" for start, end in video_ranges]

    print(limits)
    
    args = [(gpu_ids[i % num_gpus], limits[i], config_path) for i in range(total_processes)]
    
    with multiprocessing.Pool(processes=total_processes) as pool:
        pool.map(run_command, args)

if __name__ == "__main__":
    main()
