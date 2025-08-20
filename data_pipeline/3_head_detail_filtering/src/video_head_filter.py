import json
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from pathlib import Path
import os
import tqdm
import traceback
from pic_head_base_fun import (
    calculate_head_movement,
    calculate_face_rotation,
    check_face_completeness,
    calculate_face_resolution_score,
    calculate_head_orientation_score
)
import torch    
import time  
import argparse
import glob
import yaml


import sys
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+



#####################################################
def extract_ids(json_path): 
    """
    Reads a JSON file line by line, extracts all 'id' fields, and returns a list of these IDs.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        list: A list containing all the IDs.
    """
    ids = []
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                ids.append(entry['id'])
    return ids

#####################################################

class VideoFaceProcessor:
    def __init__(self, limit_videos=10, config_path='config/head_threshold.yml'):
        if torch.cuda.is_available():
            self.app = FaceAnalysis(providers=['CUDAExecutionProvider'])
            # print("using cuda")
        else:
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            # print("using cpu")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Load thresholds from config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['head_filter']
            self.score_thresholds = config['thresholds']
        
        jsonl_path = config['input_path']
        self.video_data = []
        with open(jsonl_path, 'r') as f:
            # for line in f:
            #     self.video_data.append(json.loads(line.strip()))
            self.video_data = json.load(f)
        # print(f"read {len(self.video_data)} videos")
        
        
        self.processing_stats = {
            'video_durations': {},  # video duration
            'processing_times': {}  # processing time
        }

        print(f"limit_videos is {limit_videos}")
        if isinstance(limit_videos,int):
            self.video_data = self.video_data[:limit_videos]
            breakpoint()
        elif (isinstance(limit_videos,list) or isinstance(limit_videos,tuple)) and len(limit_videos) == 2:
            self.video_data = self.video_data[limit_videos[0]:limit_videos[1]]

            #################################
            
            config_name = os.path.basename(config_path)[:-5] 
            if "_av1" in config_name:
                suffix = "_av1"
                start_end = config_name[7:-4]
                if start_end == "previous":
                    start_end = "-1_-1"
            else:
                suffix = ""
                start_end = config_name[7:]
                if start_end == "previous":
                    start_end = "-1_-1"
                
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            tmp_folder = f"{current_dir}/../log/tmp_{start_end}{suffix}"
            if not os.path.exists(tmp_folder):
                os.makedirs(tmp_folder)

            
            tmp_file = os.path.join(tmp_folder, f"video_evaluation_results_{limit_videos[0]},{limit_videos[1]}.jsonl")
            exist_ids = extract_ids(tmp_file)
            self.video_data = [item for item in self.video_data if item['id'] not in exist_ids]
            print(f"chunk {limit_videos[0]}_{limit_videos[1]} has {len(self.video_data)} left")

            ##################################
        
        print(f"process {len(self.video_data)} videos")


    def process_video(self, video_item):
        start_time = time.time()
        debug_infos = []
        video_path = video_item['video-path']
        video_id = video_item['id']
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        frames = []
        faces_sequence = []
        face_counts = []  # Track number of faces in each frame
        
        # Sample one frame per second
        for sec in range(int(duration)):
            frame_idx = sec * fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                faces = self.app.get(frame)
                face_counts.append(len(faces))  # Record face count
                
                if len(faces) == 0:
                    debug_infos.append(f"video {video_id} has no faces at second {sec}")
                elif len(faces) > 1:
                    debug_infos.append(f"video {video_id} has {len(faces)} faces at second {sec}")
                else:
                    frames.append(frame)
                    faces_sequence.append(faces)

        if not cap.isOpened():
            print(f"failed to open video: {video_path}")
            return None
        cap.release()

        if len(frames) == 0:
            return {
                'debug_infos': ['No frame with only one face detected'],
                'evaluation': None
            }
        
        if not faces_sequence:  # No faces detected in any frame
            return {
                'debug_infos': ['have faces but no faces detected in any frame'],
                'evaluation': None
            }

        rtn_dic = {
            'debug_infos': debug_infos,
            'evaluation': self.evaluate_faces(
                faces_sequence=faces_sequence, 
                sample_frame=frames[0],
                face_counts=face_counts  # Pass face counts to evaluation
            )
        }
        
        
        self.processing_stats['processing_times'][video_id] = time.time() - start_time
        self.processing_stats['video_durations'][video_id] = duration
        
        return rtn_dic

    def evaluate_faces(self, faces_sequence, sample_frame, face_counts):
        """
        Evaluate face quality metrics from a sequence of frames
        
        Args:
            faces_sequence: List of detected faces for each frame
            sample_frame: Reference frame for calculations
            face_counts: Number of faces detected in each frame
        
        Returns:
            Dictionary containing quality scores and pass/fail status
        """
        if not faces_sequence:
            return None

        # calculate head movement and face rotation
        head_movement_dic = calculate_head_movement([seq[0] for seq in faces_sequence if seq], sample_frame.shape)
        face_rotation_dic = calculate_face_rotation([seq[0] for seq in faces_sequence if seq])
        
        # calculate completeness, resolution, and orientation scores
        completeness_scores = [check_face_completeness(seq[0], sample_frame.shape) for seq in faces_sequence if seq]
        resolution_scores = [calculate_face_resolution_score(seq[0], sample_frame.shape) for seq in faces_sequence if seq]
        orientation_scores = [abs(calculate_head_orientation_score(seq[0])) for seq in faces_sequence if seq]
        
        scores = {
            # Movement and Orientation scores 
            'avg_movement': 100-100*head_movement_dic['avg_movement'],
            'min_movement': 100-100*head_movement_dic['max_movement'],
            'min_movement_id': head_movement_dic['min_movement_id'],
            
            # Rotation scores
            'avg_rotation': 100-face_rotation_dic['avg_rotation'],
            'min_rotation': 100-face_rotation_dic['min_rotation'],
            'min_rotation_id': face_rotation_dic['min_rotation_id'],
            
            # Completeness scores
            'avg_completeness': np.mean(completeness_scores),
            'min_completeness': np.min(completeness_scores),
            'min_completeness_id': int(np.argmin(completeness_scores)),
            
            # Resolution scores
            'avg_resolution': 30*np.mean(resolution_scores),
            'min_resolution': 30*np.min(resolution_scores),
            'min_resolution_id': int(np.argmin(resolution_scores)),
            
            # Orientation scores
            'avg_orientation': 100-np.mean(orientation_scores),
            'min_orientation': 100-np.max(orientation_scores),
            'min_orientation_id': int(np.argmax(orientation_scores)),  
            
            # Face consistency remains the same
            'face_consistency': 100 if all(count == 1 for count in face_counts) else max(0, 100 - 20 * (sum(count != 1 for count in face_counts))),
        }
        
        passed = all(scores[key] >= self.score_thresholds[key] for key in self.score_thresholds)
        
        return {
            'scores': scores,
            'passed': passed
        }
    def process_all_videos(self,output_path):
        results = {}
        multiface_debug_infos = []
        
        for video_item in tqdm.tqdm(self.video_data, desc="processing videos"):
            # print(f"processing video: {video_item['id']}")
            try:
                process_result = self.process_video(video_item)
                result, debug_info = process_result['evaluation'], process_result['debug_infos']

                if result:
                    for key in result['scores']:
                        result['scores'][key] = float(result['scores'][key])  # convert numpy.float32 to Python float
                    
                    results[video_item['id']] = {
                        'evaluation': result,
                        'file_info': video_item,
                    }
                if debug_info:
                    multiface_debug_infos.append(debug_info)
            except Exception as e:
                result=None
                debug_info=f"error processing video {video_item['id']}: {e}, traceback: {traceback.format_exc()}"
                
                for key in result['scores']:
                    result['scores'][key] = float(result['scores'][key])  # convert numpy.float32 to Python float
                    
                results[video_item['id']] = {
                        'evaluation': result,
                        'file_info': video_item,
                    }
                multiface_debug_infos.append(debug_info)

            if use_folder_to_save_results:
                results_path = os.path.join(output_path,f"video_evaluation_results_{file_name_prefix}.jsonl")
                errors_path = os.path.join(output_path,f"video_face_detection_errors_{file_name_prefix}.jsonl")
            else:
                results_path = os.path.join(output_path,"video_evaluation_results.jsonl")
                errors_path = os.path.join(output_path,"video_face_detection_errors.jsonl")

            with open(results_path, 'a', encoding='utf-8') as f:
                video_item['head_detail'] = result
                f.write(json.dumps(video_item, ensure_ascii=False) + "\n")
            
            if debug_info:
                with open(errors_path, 'a') as f:
                    json_str = json.dumps({video_item['id']:
                                            {
                                                'debug_info': debug_info,
                                                'file_info': video_item,
                                            }}, ensure_ascii=False)
                    f.write(json_str + "\n")

        # Save data
        if use_folder_to_save_results:
            stats_path = os.path.join(output_path, "processing_stats_"+file_name_prefix+".json")
        else:
            stats_path = os.path.join(output_path, "processing_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            stats = {
                'total_duration': sum(self.processing_stats['video_durations'].values()),
                'total_processing_time': sum(self.processing_stats['processing_times'].values()),
                'per_video_stats': {
                    video_id: {
                        'duration': self.processing_stats['video_durations'][video_id],
                        'processing_time': self.processing_stats['processing_times'][video_id]
                    } for video_id in self.processing_stats['video_durations']
                }
            }
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # analysis processing_stats
        self.analysis_processing_stats(stats,output_path)

        return results, multiface_debug_infos
    
    def analysis_processing_stats(self, stats, output_path):
        # draw a scatter plot of video_durations and processing_times
        import matplotlib.pyplot as plt
        plt.scatter(self.processing_stats['video_durations'].values(), self.processing_stats['processing_times'].values())
        plt.xlabel('video_durations')
        plt.ylabel('processing_times')
        plt.title('video_durations and processing_times')
        if use_folder_to_save_results:
            plt.savefig(os.path.join(output_path,'video_durations_and_processing_times_'+file_name_prefix+'.png'))
        else:
            plt.savefig(os.path.join(output_path,'video_durations_and_processing_times.png'))
        plt.show()



def main(limit_videos:int | tuple[int,int],
         use_folder_to_save_results:bool,
         file_name_prefix:str,
        #  json_path:str,
         config_path:str):
    
    
    # processor = VideoFaceProcessor(json_path, limit_videos=None)
    processor = VideoFaceProcessor(limit_videos=limit_videos,config_path=config_path)

    

    if use_folder_to_save_results is False:
        output_path="./log"
    elif use_folder_to_save_results is True:
        # save part jsons to a folder, 
        # output_path = os.path.join("./log",'tmp_use_folder_to_save_results') 
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)['head_filter']
        start_sample = config['start_sample']
        end_sample = config['end_sample']
        av1_flg = config['av1_flg']  
        suffix = "_av1" if av1_flg else ""
        output_path = os.path.join("./log",f"tmp_{start_sample}_{end_sample}{suffix}") 
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    results, multiface_debug_infos = processor.process_all_videos(output_path)

    passed_videos = 0
    for r in results.values():
        try:
            if r['evaluation'] and r['evaluation']['passed']:
                passed_videos += 1
        except (TypeError, KeyError):
            continue
    
    total_videos = len(results)
    print(f"\nprocess finished:")
    print(f"total videos: {total_videos}")
    print(f"passed videos: {passed_videos}")
    print(f"passed rate: {passed_videos/total_videos*100:.2f}%")
    print(f"detailed results saved to: {output_path}")


    

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_folder_to_save_results', type=bool, default=True)
    parser.add_argument('--limit_videos', type=str, default='(0,10)')
    parser.add_argument('--config_path', type=str, default='../config/config.yaml') 
    
    


    args = parser.parse_args()
    use_folder_to_save_results = args.use_folder_to_save_results
    file_name_prefix = args.limit_videos
    
    main(limit_videos=eval(args.limit_videos),use_folder_to_save_results=use_folder_to_save_results,file_name_prefix=file_name_prefix,
         config_path=args.config_path)

    print(f"total time: {round(time.time()-start_time,2)}")

