# 5. Head Quality Filtering

## Environment Setup

tested on:
- Python 3.10.14
- CUDA 12.2
- cuDNN 9.6.0

``` bash
cd Talking-Face-Datapipe/3_head_detail_filtering/
CONDA_LIB_PATH=./lib
# you can store files like libcudnn_adv.so.9.6.0 in ./lib
export LD_LIBRARY_PATH=$CONDA_LIB_PATH:$LD_LIBRARY_PATH
```

```bash
# create a new one
conda env create -f env_head.yml
conda activate env_head
```




#### Algorithm Explanation

**Algorithm Introduction**:
To evaluate the face quality in videos, scores are given based on five dimensions, with each dimension having a score range of 0 - 100 points. The higher the score, the better the quality. The specific scoring dimensions are as follows:

1. **Movement Score**:
    - Calculation method: `100 - 100 * avg_movement`
    - Calculate the average displacement of face key points between adjacent frames and normalize it by dividing by the smaller value of the image width and height.
    - Threshold requirements: Average score ≥ 80, minimum score ≥ 60.
    - The smaller the displacement, the higher the score, reflecting the stability of face movement.

2. **Orientation Score**:
    - Calculation method: `100 - sqrt(pitch_score² + yaw_score² + roll_score²)`
    - Each angle score = `|angle| / 180 * 100`
    - Threshold requirements: Average score ≥ 70, minimum score ≥ 30.
    - It reflects the degree of deviation of the face orientation from facing the camera in the current frame. The higher the score, the smaller the deviation.

3. **Completeness Score**:
    - Score based on the positions of face key points, including three area weights:
        * Eye key points (2): Total weight 0.3
        * Nose key point: Weight 0.4
        * Mouth key points (2): Total weight 0.3
    - Calculate whether the key points in each area are within the image range. If yes, the score is 1; if not, the score is 0. Sum the products of the scores and the weights (note that even if occluded (such as by a mask or on the side), as long as it does not exceed the image box, it is judged as 1).
    - Threshold requirements: Average score = 100, minimum score = 100.
    - The higher the score, the higher the face completeness.

4. **Resolution Score**:
    - Calculation method: `30 * (face_area / total_image_area)*100`
    - Calculate the proportion of the face bounding box area to the entire image area.
    - Threshold requirements: Average score ≥ 50, minimum score ≥ 40.
    - The higher the score (which can exceed 100), the higher the proportion of the face size in the frame.

5. **Rotation Score**:
    - Calculation method: `100 - avg_rotation_amplitude`
    - Calculate the change amplitude of the face's three - dimensional orientation between adjacent frames: `sqrt(Δpitch² + Δyaw² + Δroll²)`
    - Threshold requirements: Average score ≥ 70, minimum score ≥ 60.
    - The smaller the orientation change, the higher the score, reflecting the stability of the face orientation.

Additional requirements:
- Face consistency requirement: ≥ 80 points. If there is exactly one face in all frames, the score is 100. Otherwise, for each frame with a non - single face (0 or ≥ 2), 20 points are deducted, with a minimum score of 0.
- All indicators need to meet the corresponding threshold requirements to pass the screening.


## Usage

### Parameters
- `use_folder_to_save_results`: Enable batch processing with separate result files
- `limit_videos`: Process specific video range (format: "start,end")
- `CUDA_VISIBLE_DEVICES`: Specify GPU device

### Basic Usage
```bash
cd /your-base-folder/3_head_detail_filtering

bash ./head_filter.sh
```


## Output 
json file: /your-base-folder/outputs/head/filtered_videos_0_1000.json (Here we take 0-1000 samples as an example. You can change the start_sample and end_sample to fit your own data.).

### Success Cases
```json
{
    "video_id": {
        "evaluation": {
            "scores": {
                "movement": 99.97,
                "orientation": 87.72,
                "completeness": 100.0,
                "resolution": 100.0,
                "rotation": 93.72
            },
            "passed": true
        },
        "file_info": {
            "video-path": "path/to/video",
            "video-id": "id",
            "audio-path": "path/to/audio"
        }
    }
}
```


