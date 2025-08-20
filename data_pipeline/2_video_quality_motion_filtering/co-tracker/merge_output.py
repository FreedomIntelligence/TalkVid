import json
import os

def merge_json_files(output_files, final_output):
    """
    Merges all output_files (sub-results) into a single final_output file.
    It is assumed that all sub-results are structured as a list of dicts.
    If more complex processing is needed, such as deduplication or merging logic based on an ID,
    it can be implemented here.
    """
    merged_data = []
    for ofile in output_files:
        if not os.path.exists(ofile):
            continue
        with open(ofile, "r", encoding="utf-8") as f:
            part = []
            for line in f:
                part.append(json.loads(line))

            merged_data.extend(part)

    # If deduplication based on "id" is needed, additional processing can be done here.
    # e.g., use a dictionary to store id -> item

    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    pass
