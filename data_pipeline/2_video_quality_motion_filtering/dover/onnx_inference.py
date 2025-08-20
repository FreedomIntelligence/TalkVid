import torch
import argparse
import pickle as pkl
import decord
import numpy as np
import yaml
import onnxruntime as ort
import json
import os
import time
import torch.multiprocessing as mp
from tqdm import tqdm

from heapq import heappush, heappop
from typing import List, Dict

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)

def fuse_results(results: list):
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (
        results[1] + 0.08285
    ) / 0.03774 * 0.3896
    return 1 / (1 + np.exp(-x))

def gaussian_rescale(pr):
    pr = (pr - np.mean(pr)) / np.std(pr)
    return pr

def uniform_rescale(pr):
    return np.arange(len(pr))[np.argsort(pr).argsort()] / len(pr)

def rescale_results(results: list, vname="undefined"):
    dbs = {
        "livevqc": "LIVE_VQC",
        "kv1k": "KoNViD-1k",
        "ltest": "LSVQ_Test",
        "l1080p": "LSVQ_1080P",
        "ytugc": "YouTube_UGC",
    }
    scores = {}
    total_tech_percentile = 0
    total_tech_norm_score = 0
    total_aes_percentile = 0
    total_aes_norm_score = 0
    
    for abbr, full_name in dbs.items():
        with open(f"dover_predictions/val-{abbr}.pkl", "rb") as f:
            pr_labels = pkl.load(f)
        aqe_score_set = pr_labels["resize"]
        tqe_score_set = pr_labels["fragments"]
        tqe_score_set_p = np.concatenate((np.array([results[0]]), tqe_score_set), 0)
        aqe_score_set_p = np.concatenate((np.array([results[1]]), aqe_score_set), 0)
        tqe_nscore = gaussian_rescale(tqe_score_set_p)[0]
        tqe_uscore = uniform_rescale(tqe_score_set_p)[0]
        aqe_nscore = gaussian_rescale(aqe_score_set_p)[0]
        aqe_uscore = uniform_rescale(aqe_score_set_p)[0]
        
        total_tech_percentile += tqe_uscore
        total_tech_norm_score += tqe_nscore
        total_aes_percentile += aqe_uscore
        total_aes_norm_score += aqe_nscore
        
        scores[full_name] = {
            "technical_quality": {
                "percentile": int(tqe_uscore*100),
                "normalized_score": float(f"{tqe_nscore:.2f}")
            },
            "aesthetic_quality": {
                "percentile": int(aqe_uscore*100),
                "normalized_score": float(f"{aqe_nscore:.2f}")
            }
        }
    
    # calculate average scores across all databases
    num_dbs = len(dbs)
    scores["overall"] = {
        "technical_quality": {
            "percentile": int((total_tech_percentile/num_dbs)*100),
            "normalized_score": float(f"{(total_tech_norm_score/num_dbs):.2f}")
        },
        "aesthetic_quality": {
            "percentile": int((total_aes_percentile/num_dbs)*100),
            "normalized_score": float(f"{(total_aes_norm_score/num_dbs):.2f}")
        }
    }
    
    return scores

def process_video(video_info, ort_session, output_path):
    video_id = video_info['id']
    video_path = video_info['video-path']
    
    try:
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
        
        with open("dover.yml", "r") as f:
            opt = yaml.safe_load(f)
        
        dopt = opt["data"]["val-l1080p"]["args"]
        
        temporal_samplers = {}
        for stype, sopt in dopt["sample_types"].items():
            if "t_frag" not in sopt:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )
        
        views, _ = spatial_temporal_view_decomposition(
            video_path, dopt["sample_types"], temporal_samplers
        )
        
        for k, v in views.items():
            num_clips = dopt["sample_types"][k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
            )
        
        aes_input = views["aesthetic"]
        tech_input = views["technical"]
        
        start_time = time.time()
        predictions = ort_session.run(None, {
            "aes_view": aes_input.numpy(),
            "tech_view": tech_input.numpy()
        })
        inference_time = time.time() - start_time
        
        scores = [np.mean(s) for s in predictions]
        fused_score = fuse_results(scores)
        detailed_scores = rescale_results(scores, video_id)
        video_info['detailed_scores'] = detailed_scores
        
        output_data = video_info
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        return True, inference_time
        
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return False, 0

def process_video_batch(gpu_id, video_infos, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    providers = ['CUDAExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    ort_session = ort.InferenceSession(
        config['model']['onnx_path'], 
        providers=providers, 
        sess_options=session_options
    )
    
    total_time = 0
    processed_count = 0
    total_video_duration = 0
    
    for video_info in video_infos:
        output_filename = f"{video_info['id']}_dover.json"
        output_path = os.path.join(config['out_dir'], output_filename)
        
        if os.path.exists(output_path):
            continue
            
        success, process_time = process_video(video_info, ort_session, output_path)
        
        if success:
            total_time += process_time
            processed_count += 1
            if 'durations' in video_info:
                duration_str = video_info.get('durations').replace('s', '')
                try:
                    duration_seconds = float(duration_str)
                    total_video_duration += duration_seconds
                except ValueError:
                    print(f"Can't get video duration: {video_info.get('durations')}")
            elif 'duration' in video_info:
                duration_seconds = video_info.get('duration')
                total_video_duration += duration_seconds
            else:
                print(f"No Video duraiton: {video_info}")
            
            print(f"Process {gpu_id} deals {video_info['id']} for: {process_time:.2f}s")
    
    return {
        'total_time': total_time,
        'processed_count': processed_count,
        'total_video_duration': total_video_duration
    }

def skip_successful_video(video_infos, directory): 
    """
    Skip the video that has been processed successfully
    """
    video_left = []
    for item in video_infos:
        file_path = os.path.join(directory, f"{item['id']}_dover.json")
        if os.path.exists(file_path) != True:
            video_left.append(item)
    return video_left

def parse_duration(duration_str: str) -> float:
    return float(duration_str.rstrip('s'))

def balanced_partition(data: List[Dict], num_process: int) -> List[List[Dict]]:
    """
    Divide data into num_process groups, each group has approximately equal total duration.
    
    :param data: List[Dict],key for "durations" with value "7.500s" 
    :param num_process: int,chunks number
    :return: List[List[Dict]],divided data
    """
    
    data.sort(key=lambda x: parse_duration(x["durations"]), reverse=True)
    
    heap = [(0, i, []) for i in range(num_process)]
    
    for item in data:
        total_duration, index, group = heappop(heap)
        duration_value = parse_duration(item["durations"])
        group.append(item)
        heappush(heap, (total_duration + duration_value, index, group))
    
    result = [[] for _ in range(num_process)]
    for _, index, group in heap:
        result[index] = group
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    total_wall_time_start = time.time()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['dover_filter'] # Original 
    
    os.makedirs(config['out_dir'], exist_ok=True)
    
    with open(config['input_path'], 'r') as f:
        video_infos = json.load(f)
    
    directory = config['out_dir']
    video_infos = skip_successful_video(video_infos, directory)
    print(f"Left {len(video_infos)} videos to process")
    ##########
    
    if config['test_num'] > 0:
        video_infos = video_infos[:config['test_num']]
        print(f"Dealing {len(video_infos)} videos...")
    
    num_gpus = len(config['cuda']['devices'])
    processes_per_gpu = config['num_processes']
    total_processes = num_gpus * processes_per_gpu
    
    print(f"Use {num_gpus} GPU, each GPU has {processes_per_gpu} process, in total {total_processes} processes")
    
    num_videos = len(video_infos)
    batch_size = num_videos // total_processes

    video_batches = balanced_partition(video_infos, total_processes)

    for id, batch in enumerate(video_batches):
        batch_duraiton = 0
        for item in batch:
            batch_duraiton += parse_duration(item["durations"])
        print(f"batch{id} has {len(batch)} videos, total duration is {batch_duraiton}")


    if len(video_batches) > total_processes:
        video_batches[-2].extend(video_batches[-1])
        video_batches.pop()
    
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=total_processes)
    
    process_args = []
    for i, batch in enumerate(video_batches):
        gpu_idx = i % num_gpus
        gpu_id = config['cuda']['devices'][gpu_idx]
        process_args.append((gpu_id, batch, config))
    
    results = pool.starmap(process_video_batch, process_args)
    
    pool.close()
    pool.join()
    
    total_time = sum(r['total_time'] for r in results)
    total_processed = sum(r['processed_count'] for r in results)
    total_video_duration = sum(r['total_video_duration'] for r in results)
    
    if total_processed > 0:
        avg_time = total_time / total_processed
        total_wall_time = time.time() - total_wall_time_start
        
        print(f"\nProcessing Statistics:")
        print(f"Total videos processed: {total_processed}")
        print(f"Total CPU processing time: {total_time:.2f} seconds")
        print(f"Total wall-clock time elapsed: {total_wall_time:.2f} seconds")
        print(f"Average processing time per video: {avg_time:.2f} seconds")
        print(f"Total video duration processed: {total_video_duration:.2f} seconds ({total_video_duration/60:.2f} minutes)")
        if total_video_duration > 0:
            acceleration_ratio = total_video_duration / total_wall_time
            print(f"Acceleration ratio: {acceleration_ratio:.2f}x")
            print(f"Parallel efficiency: {(total_time/total_processes)/total_wall_time:.2f}")
        else:
            print("Cannot calculate acceleration ratio because the total video duration is 0")
    
    result_files = [f for f in os.listdir(config['out_dir']) if f.endswith('_dover.json')]
    merged_results = []
    
    for result_file in result_files:
        with open(os.path.join(config['out_dir'], result_file), 'r') as f:
            merged_results.append(json.load(f))
    
    #filter results based on the dover scores
    for result in merged_results:
        if result['detailed_scores']['overall']['technical_quality']['normalized_score']  < config['threshold']:
            result['pass_video_quality'] = False
        else:
            result['pass_video_quality'] = True
        result['dover_scores'] = result['detailed_scores']['overall']['technical_quality']['normalized_score']
        del result['detailed_scores']
    

    start_sample = config['start_sample']
    end_sample = config['end_sample']
    av1_flag = config['av1_flg']
    suffix = "_av1" if av1_flag else ""
    merged_json_path = os.path.join(os.path.dirname(config['out_dir']), f"merged_dover_{str(start_sample)}_{str(end_sample)}{suffix}.json")
    if start_sample == 'previous':
        merged_json_path = os.path.join(os.path.dirname(config['out_dir']), f"merged_dover_previous{suffix}.json")

    with open(merged_json_path, 'w') as f:
        json.dump(merged_results, f, indent=4, ensure_ascii=False)
        
   
   
    pass_results = [result for result in merged_results if result['pass_video_quality']]

    filtered_json_path = os.path.join(os.path.dirname(config['out_dir']), f"dover_success_{str(start_sample)}_{str(end_sample)}{suffix}.json")
    if start_sample == 'previous':
        filtered_json_path = os.path.join(os.path.dirname(config['out_dir']), f"dover_success_previous{suffix}.json")
    with open(filtered_json_path, 'w') as f:
        json.dump(pass_results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
        
