import json
import re
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

from utils.utils import parse_args, save_json


def read_process_number(file_path):
    '''
    Read the maximum process number from a file.
    
    :param file_path: The path to the file containing the maximum number.
    :return: The maximum number as an integer, or None if an error occurs.
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the content, assuming it contains only the maximum number as a string
            process_number = file.read().strip()  # Remove any leading/trailing whitespace
            # Convert the string to an integer
            process_number = int(process_number)
            return process_number
    except FileNotFoundError:
        print(f"File not found: {file_path}, aggregation from scratch")
        # Get the directory path from the file path
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Write the process_number to the output file
        process_number = 0
        with open(file_path, 'w') as f:
            f.write(str(process_number))
        return process_number
    except ValueError:
        print(f"The file does not contain a valid integer.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_process_list(process_file_folder, min_num=0):
    # Define the pattern to match the number in the filename
    pattern = re.compile(r'\d+')  # This pattern matches one or more digits
    process_file_list = []

    # Loop through all the files in the specified directory
    for filename in os.listdir(process_file_folder):
        # Check if the filename starts with the specified prefix and ends with '.json'
        if filename.startswith('all_scenes_data_classification') and filename.endswith('.json'):
            # Use the pattern to search for digits in the filename
            match = pattern.search(filename)
            if match:
                # Extract the number from the match object and convert it to an integer
                num = int(match.group())
                # If the number is greater than max_num, add the filename to the list
                if num > min_num:
                    process_file_list.append(filename)
    
    return process_file_list


def calculate_duration(sample):
    return sample['end-time'] - sample['start-time']


def categorize_temporal(short_data, long_data, aggregation_data, add_data, duration_threshold=5):

    for sample in add_data:
        aggregation_data.append(sample)
        duration = calculate_duration(sample)
        if duration <= duration_threshold:
            short_data.append(sample)
        elif duration > duration_threshold:
            long_data.append(sample)

    return short_data, long_data, aggregation_data


if __name__ == "__main__":
    args = parse_args()

    short_json_path = args.short_json_path
    long_json_path = args.long_json_path
    aggregation_json_path = args.aggregation_json_path
    duration_threshold = args.duration_threshold
    if os.path.isfile(short_json_path):
        with open(short_json_path, 'r', encoding='utf-8') as f:
            short_data = json.load(f) # complete or only one
    else:
        short_data = []
    if os.path.isfile(long_json_path):
        with open(long_json_path, 'r', encoding='utf-8') as f:
            long_data = json.load(f) # complete or only one
    else:
        long_data = []
    if os.path.isfile(aggregation_json_path):
        with open(aggregation_json_path, 'r', encoding='utf-8') as f:
            aggregation_data = json.load(f) # complete or only one
    else:
        aggregation_data = []

    max_ckpt_file = args.aggregation_max_ckpt_path

    process_number = read_process_number(max_ckpt_file)

    # Get the directory path from the file path
    directory = os.path.dirname(max_ckpt_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    process_file_list = get_process_list(args.output_json_folder, process_number)
    first_flag = True # first json file need check
    for process_file in process_file_list:
        part_json_path = os.path.join(args.output_json_folder, process_file)
        if first_flag and process_number != 0:
            existing_ids = set()
            try:
                with open(part_json_path, 'r', encoding='utf-8') as f:
                    temp_data = json.load(f)
            except json.JSONDecodeError as e:
                with open(part_json_path, 'r', encoding='utf-8') as f:
                    try:
                        temp_data = []
                        for line in f:
                            sample = json.loads(line)
                            temp_data.append(sample)
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        exit()
            except Exception as e:
                print(f"An error occurred: {e}")
                exit()
            for sample in aggregation_data:
                existing_ids.add(sample['id'])
            temp_data = [sample for sample in temp_data if sample['id'] not in existing_ids]
            short_data, long_data, aggregation_data = categorize_temporal(short_data, long_data, aggregation_data, temp_data, duration_threshold=duration_threshold)
            first_flag = False
            process_number = process_number + 1
        else:
            try:
                with open(part_json_path, 'r', encoding='utf-8') as f:
                    temp_data = json.load(f)
            except json.JSONDecodeError as e:
                with open(part_json_path, 'r', encoding='utf-8') as f:
                    try:
                        temp_data = []
                        for line in f:
                            sample = json.loads(line)
                            temp_data.append(sample)
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        exit()
            except Exception as e:
                print(f"An error occurred: {e}")
                exit()
            short_data, long_data, aggregation_data = categorize_temporal(short_data, long_data, aggregation_data, temp_data, duration_threshold=duration_threshold)
            process_number = process_number + 1

    # Get the directory path from the file path
    directory = os.path.dirname(max_ckpt_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Write the max_number to the output file
    with open(max_ckpt_file, 'w') as f:
        f.write(str(process_number))
    directory = os.path.dirname(aggregation_json_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_json(short_data, short_json_path)
    save_json(long_data, long_json_path)
    save_json(aggregation_data, aggregation_json_path)
    print('aggregation complete')
        