# 3. Video Quality and Motion Filtering

## 3.1 DOVER
# DOVER

Official Code for [ICCV2023] Paper *"Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives"*. 
Official Code, Demo, Weights for the [Disentangled Objective Video Quality Evaluator (DOVER)](arxiv.org/abs/2211.04894).

- 22 Nov, 2023: We upload weights of [DOVER](https://huggingface.co/teowu/DOVER/resolve/main/DOVER.pth?download=true) and [DOVER++](https://huggingface.co/teowu/DOVER/resolve/main/DOVER_plus_plus.pth?download=true) to Hugging Face models.
- 21 Nov, 2023: The release note of [DIVIDE database](get_divide_dataset/) is updated. 
- 1 Aug, 2023: ONNX conversion script for DOVER has been released. Short tip: after installation, run [this](https://github.com/VQAssessment/DOVER/blob/master/convert_to_onnx.py) and then [this](https://github.com/VQAssessment/DOVER/blob/master/onnx_inference.py).
- 17 Jul, 2023: DOVER has been accepted by ICCV2023.
- 9 Feb, 2023: **DOVER-Mobile** is available! Evaluate on CPU with Very High Speed!
- 16 Jan, 2023: Full Training Code Available (include LVBS). See below.
- 10 Dec, 2022: Now the evaluation tool can directly predict a fused score for any video. See [here](https://github.com/QualityAssessment/DOVER#new-get-the-fused-quality-score-for-use).


![visitors](https://visitor-badge.laobi.icu/badge?page_id=teowu/DOVER) [![](https://img.shields.io/github/stars/QualityAssessment/DOVER)](https://github.com/QualityAssessment/DOVER)
[![State-of-the-Art](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/QualityAssessment/DOVER)
<a href="https://colab.research.google.com/github/taskswithcode/DOVER/blob/master/TWCDOVER.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 



**DOVER** Pseudo-labelled Quality scores of [Kinetics-400](https://www.deepmind.com/open-source/kinetics): [CSV](https://github.com/QualityAssessment/DOVER/raw/master/dover_predictions/kinetics_400_1.csv)

**DOVER** Pseudo-labelled Quality scores of [YFCC-100M](http://projects.dfki.uni-kl.de/yfcc100m/): [CSV](https://github.com/QualityAssessment/DOVER/raw/master/dover_predictions/yfcc_100m_1.csv)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangling-aesthetic-and-technical-effects/video-quality-assessment-on-konvid-1k)](https://paperswithcode.com/sota/video-quality-assessment-on-konvid-1k?p=disentangling-aesthetic-and-technical-effects)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangling-aesthetic-and-technical-effects/video-quality-assessment-on-live-fb-lsvq)](https://paperswithcode.com/sota/video-quality-assessment-on-live-fb-lsvq?p=disentangling-aesthetic-and-technical-effects)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangling-aesthetic-and-technical-effects/video-quality-assessment-on-live-vqc)](https://paperswithcode.com/sota/video-quality-assessment-on-live-vqc?p=disentangling-aesthetic-and-technical-effects)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangling-aesthetic-and-technical-effects/video-quality-assessment-on-youtube-ugc)](https://paperswithcode.com/sota/video-quality-assessment-on-youtube-ugc?p=disentangling-aesthetic-and-technical-effects)


![Fig](figs/in_the_wild_on_kinetics.png)

Corresponding video results can be found [here](https://github.com/QualityAssessment/DOVER/tree/master/figs).




## Introduction


*In-the-wild UGC-VQA is entangled by aesthetic and technical perspectives, which may result in different opinions on the term **QUALITY**.*

![Fig](figs/problem_definition.png)

### the proposed DOVER

*This inspires us to propose a simple and effective way to disengtangle the two perspectives from **EXISTING** UGC-VQA datasets.*

![Fig](figs/approach.png)

### DOVER or DOVER-Mobile

DOVER-Mobile changes the backbone of two branches in DOVER into `convnext_v2_femto` (inflated). The whole DOVER-Mobile has only **9.86M** (5.7x less than DOVER) parameters, **52.3GFLops** (5.4x less than DOVER), and less than **1.9GB graphic memory cost** (3.1x less than DOVER) during inference.

The speed on CPU is also much faster (**1.4s** vs 3.6s per video, on our test environment).

Results comparison:
|  Metric: PLCC    | KoNViD-1k | LIVE-VQC | LSVQ_test | LSVQ_1080p | Speed on CPU |
| ----  |    ----   |   ----  |      ----     |   ----  | ---- |
| [**DOVER**](https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth) | 0.883 | 0.854 | 0.889 | 0.830 | 3.6s |
| [**DOVER-Mobile**](https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth) | 0.853 | 0.835 | 0.867 | 0.802 | **1.4s**:rocket: |
| BVQA (Li *et al*, TCSVT 2022) | 0.839 | 0.824 | 0.854 | 0.791 | >300s |
| Patch-VQ (Ying *et al, CVPR 2021) | 0.795 | 0.807 | 0.828 | 0.739 | >100s |

To switch to DOVER-Mobile, please add `-o dover-mobile.yml` at the end of any of following scripts (train, test, validate).

## Install

Important Note: 
Due to the file size limit for supplementary materials, the content provided here may be incomplete. For the full version of the repository, including all code and data, please clone the official GitHub repository using the following command:

```shell
git clone https://github.com/QualityAssessment/DOVER.git 
cd DOVER 
pip install -e .  
mkdir pretrained_weights 
cd pretrained_weights 
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth 
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth
cd ..
```

## Tip to Download LSVQ

```python
import os, glob

snapshot_download("teowu/LSVQ-videos", repo_type="dataset", local_dir="./", local_dir_use_symlinks=False)

gz_files = glob.glob("*.tar.gz")

for gz_file in gz_files:
    print(gz_file)
    os.system("tar -xzf {}".format(gz_file))
```

## Evaluation: Judge the Quality of Any Video

### New! ONNX Conversion is available

We have now supported to convert to ONNX, which can easily be deployed to a wide range of devices.

After the installation above, you can do this with a single step:

```shell
python convert_to_onnx.py
```

You may infer with ONNX as follows.

```shell
python onnx_inference.py -v ./demo/17734.mp4
```


### Try on Demos



You can run a single command to judge the quality of the demo videos in comparison with videos in VQA datasets. 

```shell
    python evaluate_one_video.py -v ./demo/17734.mp4 -f
```

or 

```shell
    python evaluate_one_video.py -v ./demo/1724.mp4 -f
```

You can also remove the `-f` to get the relative aesthetic and technical ranks in common UGC-VQA databases. 


### Evaluate on your customized videos


Or choose any video you like to predict its quality:


```shell
    python evaluate_one_video.py -v $YOUR_SPECIFIED_VIDEO_PATH$ -f
```

### Outputs

#### ITU-Standarized Overall Video Quality Score

The script can directly score the video's overall quality (considering both perspectives) between [0,1].

```shell
    python evaluate_one_video.py -v $YOUR_SPECIFIED_VIDEO_PATH$
```

The final output score is normalized and converted via ITU standards.

#### Old-Behaviour: Relative Aesthetic/Technical Ranks

You should get some outputs as follows if you remove the `-f`. As different datasets have different scales, an absolute video quality score is useless, but the comparison on both **aesthetic** and **technical quality** between the input video and all videos in specific sets are good indicators for how good the quality of the video is.

In the current version, you can get the analysis of the video's quality as follows (the normalized scores are following `N(0,1)`, so you can expect scores > 0 are related to better quality).


```
Compared with all videos in the LIVE_VQC dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 43% of videos, with normalized score 0.10.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 64% of videos, with normalized score 0.51.
Compared with all videos in the KoNViD-1k dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 75% of videos, with normalized score 0.77.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 91% of videos, with normalized score 1.21.
Compared with all videos in the LSVQ_Test dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 69% of videos, with normalized score 0.59.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 79% of videos, with normalized score 0.85.
Compared with all videos in the LSVQ_1080P dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 53% of videos, with normalized score 0.25.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 54% of videos, with normalized score 0.25.
Compared with all videos in the YouTube_UGC dataset:
-- the technical quality of video [./demo/17734.mp4] is better than 71% of videos, with normalized score 0.65.
-- the aesthetic quality of video [./demo/17734.mp4] is better than 80% of videos, with normalized score 0.86.
```


## Evaluate on a Set of Unlabelled Videos


```shell
    python evaluate_a_set_of_videos.py -in $YOUR_SPECIFIED_DIR$ -out $OUTPUT_CSV_PATH$
```

The results are stored as `.csv` files in dover_predictions in your `OUTPUT_CSV_PATH`.

Please feel free to use DOVER to pseudo-label your non-quality video datasets.


## Data Preparation

We have already converted the labels for most popular datasets you will need for Blind Video Quality Assessment,
and the download links for the **videos** are as follows:

:book: LSVQ: [Github](https://github.com/baidut/PatchVQ)

:book: KoNViD-1k: [Official Site](http://database.mmsp-kn.de/konvid-1k-database.html)

:book: LIVE-VQC: [Official Site](http://live.ece.utexas.edu/research/LIVEVQC)

:book: YouTube-UGC: [Official Site](https://media.withyoutube.com)

*(Please contact the original authors if the download links were unavailable.)*

After downloading, kindly put them under the `../datasets` or anywhere but remember to change the `data_prefix` respectively in the [config file](dover.yml).


## Dataset-wise Default Inference

To test the pre-trained DOVER on multiple datasets, please run the following shell command:

```shell
    python default_infer.py
```

# Training: Adapt DOVER to your video quality dataset!

Now you can employ ***head-only/end-to-end transfer*** of DOVER to get dataset-specific VQA prediction heads. 

We still recommend **head-only** transfer. As we have evaluated in the paper, this method has very similar performance with *end-to-end transfer* (usually 1%~2% difference), but will require **much less** GPU memory, as follows:

```shell
    python transfer_learning.py -t $YOUR_SPECIFIED_DATASET_NAME$
```

For existing public datasets, type the following commands for respective ones:

- `python transfer_learning.py -t val-kv1k` for KoNViD-1k.
- `python transfer_learning.py -t val-ytugc` for YouTube-UGC.
- `python transfer_learning.py -t val-cvd2014` for CVD2014.
- `python transfer_learning.py -t val-livevqc` for LIVE-VQC.


As the backbone will not be updated here, the checkpoint saving process will only save the regression heads with only `398KB` file size (compared with `200+MB` size of the full model). To use it, simply replace the head weights with the official weights [DOVER.pth](https://github.com/teowu/DOVER/releases/download/v0.1.0/DOVER.pth).

We also support ***end-to-end*** fine-tune right now (by modifying the `num_epochs: 0` to `num_epochs: 15` in `./dover.yml`). It will require more memory cost and more storage cost for the weights (with full parameters) saved, but will result in optimal accuracy.

Fine-tuning curves by authors can be found here: [Official Curves](https://wandb.ai/timothyhwu/DOVER) for reference.

### 3.1.1 Install Dependencies

```bash
conda create --name dover
pip install -r requirements.txt # python dependencies

apt update
apt install -y libgl1-mesa-glx
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get install -y libcudnn9-cuda-12 && apt-get install -y libcudnn9-dev-cuda-12 # install cudnn9
```

### 3.1.2 parameter settings
input_path: Path to the input JSON file
out_dir: Output directory
test_num: Number of videos to process in test mode, 0 means process all videos
num_processes: Number of processes per GPU
threshold: Threshold, videos with a score below this value will be filtered
model:
    onnx_path: Path to the ONNX model
cuda:
    devices: List of GPU device IDs to use



### 3.1.3 Execution

```bash
conda activate dover

cd /your-base-folder/2_video_quality_motion_filtering
./video_quality_dover.sh 
```

## 3.2 Co-Tracker

# CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos

### [Project Page](https://cotracker3.github.io/) | [Paper #1](https://arxiv.org/abs/2307.07635) | [Paper #2](https://arxiv.org/abs/2410.11831) |  [X Thread](https://twitter.com/n_karaev/status/1742638906355470772) | [BibTeX](#citing-cotracker)

**CoTracker** is a fast transformer-based model that can track any point in a video. It brings to tracking some of the benefits of Optical Flow.

CoTracker can track:

- **Any pixel** in a video
- A **quasi-dense** set of pixels together
- Points can be manually selected or sampled on a grid in any video frame


## Installation Instructions
You can use a Pretrained Model via PyTorch Hub, as described above, or install CoTracker from this GitHub repo.
This is the best way if you need to run our local demo or evaluate/train CoTracker.

Ensure you have both _PyTorch_ and _TorchVision_ installed on your system. Follow the instructions [here](https://pytorch.org/get-started/locally/) for the installation.
We strongly recommend installing both PyTorch and TorchVision with CUDA support, although for small tasks CoTracker can be run on CPU.


### Install a Development Version

Important Note: 
Due to the file size limit for supplementary materials, the content provided here may be incomplete. For the full version of the repository, including all code and data, please clone the official GitHub repository using the following command:

```bash
conda create --name cotracker
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
apt install -y libgl1-mesa-glx
```

You can manually download all CoTracker3 checkpoints (baseline and scaled models, as well as single and sliding window architectures) from the links below and place them in the `checkpoints` folder as follows:

```bash
mkdir -p checkpoints
cd checkpoints
# download the online (multi window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
# download the offline (single window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
cd ..
```
You can also download CoTracker3 checkpoints trained only on Kubric:
```bash
# download the online (sliding window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/baseline_online.pth
# download the offline (single window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/baseline_offline.pth
```
For old checkpoints, see [this section](#previous-version).

## Inference with cotracker3
```bash
conda activate cotracker
cd /your-base-folder/2_video_quality_motion_filtering

./video_motion_cotracker.sh
```




