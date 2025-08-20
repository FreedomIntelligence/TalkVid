
import glob
import os
import argparse
import json
import yaml


parser = argparse.ArgumentParser(description="Run video head filter with multiprocessing.")
parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")

parser.add_argument('--log_path_to_merge', type=str, default='./log')
parser.add_argument('--filter_failed_videos', type=bool, default=False, help='if True, only keep videos passed the threshold')
# recommened use False, then use  python src/filter_out_head_videos.py to filter out head videos, keep all the results for future analysis
parser.add_argument('--output_path', type=str, default='./log')
args = parser.parse_args()
    
config_path = args.config
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)['head_filter']

start_sample = config['start_sample']
end_sample = config['end_sample']
av1_flg = config['av1_flg']  
suffix = "_av1" if av1_flg else ""

results_file_list = glob.glob(os.path.join(args.log_path_to_merge, f"tmp_{start_sample}_{end_sample}{suffix}", "video_evaluation_results_*.jsonl"))
errors_file_list = glob.glob(os.path.join(args.log_path_to_merge, f"tmp_{start_sample}_{end_sample}{suffix}", "video_face_detection_errors_*.jsonl"))


print('results_file_list',results_file_list)
print('errors_file_list',errors_file_list)


# errors_path_merge = './log/video_face_detection_errors.jsonl' # original
# results_path_merge = './log/video_evaluation_results.jsonl' # original
errors_path_merge = f"./log/video_face_detection_errors_{start_sample}_{end_sample}{suffix}.jsonl"
results_path_merge = f"./log/video_evaluation_results_{start_sample}_{end_sample}{suffix}.jsonl"

# merge video evaluation results
with open(results_path_merge, 'w', encoding='utf-8') as out_f:
    for file_path in results_file_list:
        if not args.filter_failed_videos:
            with open(file_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    out_f.write(line)
        else:
            with open(file_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if json.loads(line)['head_detail']['passed']:
                        out_f.write(line)

print('merged',results_path_merge)

# merge errors
with open(errors_path_merge, 'w', encoding='utf-8') as out_f:
    for file_path in errors_file_list:
        with open(file_path, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                out_f.write(line)

print('merged',errors_path_merge)

