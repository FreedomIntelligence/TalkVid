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

```bash
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
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
cd your-base-folder/2_video_quality_motion_filtering/co-tracker

python multi_gpu_runner.py --config config.yaml
```