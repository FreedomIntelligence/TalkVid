import json
import os
import yaml
from pathlib import Path
import argparse

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

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
########################

def filter_videos(input_jsonl, output_dir, config_path, start_sample, end_sample, suffix):
    # Load thresholds
    with open(config_path, 'r') as f:
        # config = yaml.safe_load(f) # Original
        config = yaml.safe_load(f)['head_filter']
        thresholds = config['thresholds']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"filtered_videos_{start_sample}_{end_sample}{suffix}.json") 
    
    # passed_videos = {}
    passed_videos=[]
    
    # Read and filter videos
    with open(input_jsonl, 'r') as f:
        index = 0
        for line in f:
            video_data = json.loads(line.strip())
            # print(index, ":", video_data)
            index += 1
            data = video_data['head_detail']
            if data:
                passes_threshold=data['passed']
                
                if passes_threshold:
                    passed_videos.append(line)
    
    # Write filtered results
    with open(output_path, 'w') as f:
        for line in passed_videos:
            # json_str = json.dumps({video_id: data}, ensure_ascii=False)
            f.write(line)

    convert_json_line_to_general(output_path)
    
    print(f"Total videos passed: {len(passed_videos)}")
    print(f"Filtered results saved to: {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run video head filter with multiprocessing.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")

    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)['head_filter']

    start_sample = config['start_sample']
    end_sample = config['end_sample']
    av1_flg = config['av1_flg']  
    suffix = "_av1" if av1_flg else ""

    input_jsonl = f"./log/video_evaluation_results_{start_sample}_{end_sample}{suffix}.jsonl"
    print(input_jsonl)
    output_dir = config['output_dir']

    
    
    filter_videos(input_jsonl, output_dir, config_path, start_sample, end_sample, suffix)
