# 2. Video Rough Segmentation

## 2.1 Installation

- Create a conda virtual environment and activate it:

  ```bash
  conda env create -f datapipe.yaml
  conda activate video-py310
  apt update
  apt install ffmpeg
  apt install -y libgl1-mesa-glx

## 2.2 Video Clips
This document concludes a brief introduction for Pyscenedetect toolbox [reference](https://www.scenedetect.com/)
and the usage of our clip code

#### 2.2.1 Introduction of Pyscenedetect
you can use pyscenedetect to clip your video easily. \
this project recommand two different Detectors in pyscenedetect: \
detect-content, and \
detect-hist

#### 2.2.2 Detector

##### detect-content
```python
class scenedetect.detectors.content_detector.ContentDetector(threshold=27.0, min_scene_len=15, weights=Components(delta_hue=1.0, delta_sat=1.0, delta_lum=1.0, delta_edges=0.0), luma_only=False, kernel_size=None, filter_mode=Mode.MERGE)
```
the most important parameter is threshold(Default 27.0), we recommend 10.0 in our project.

##### detect-hist
```python
class scenedetect.detectors.histogram_detector.HistogramDetector(threshold=0.05, bins=256, min_scene_len=15)
```
the most important parameter is threshold(Default 0.05), we recommend 0.085 in our project.


##### recommendation
We advice detect-content and detect-hist.

#### 2.2.3 Paramaters
- input_json_file: the data of original videos.
- video_clips_folder: the folder of video clips.
- output_json_folder: the data of video clips
- use_fixed_duration: True for using the fixed duration/ False for not using the fixed duration
- clip_duration: the length of the fixed duration
- detector_type: the type of the Pyscene detector (we recommand "Histogram" and "Content" in our project)
- detector_threshold: the threshold of the detector (we recommand 0.085 for "Histogram" and 10.0 for "Content" in our project)



#### 2.2.4 Basic Usage
```bash
conda activate video-py310
cd ./1_video_rough_segmentation
./rough_segementation.sh
```
output 
data: `os.path.join(output_json_folder,  f"_{detector_type}_threshold={detector_parameter}.json")`
video clips: `video_clips_folder`



## 2.3. Duration and Vocal Check
### 2.3.1. Duration check 
The video clips with duration shorter than the threshold (default to be 5s) will be removed.
### 2.3.2. Vocal check
The video clips without subtitle will be removed 

### 2.3.3 Parameters
- duration_threshold: The video clips with duration shorter than the threshold (default to be 5s) will be removed.

### 2.3.4  Basic Usage

Duration and Vocal will be checked when processing ./rough_segementation.sh

output: 
video clips with duration checked data: `os.path.join(output_json_folder,  f"all_scenes_data_duration_{duration_threshold}_checked.json")`
video clips with duration and vocal checked data: `os.path.join(output_json_folder,  f"all_scenes_data_duration_{duration_threshold}_vocal_checked.json")`