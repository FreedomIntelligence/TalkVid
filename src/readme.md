# TalkVid: Training & Inference

## Setup

```bash
# Create environment
conda create -n talkingface python=3.10 -y
conda activate talkingface
pip install -r requirements.txt
````

## Training

```bash
bash train_stage.sh
```

## Inference

```bash
bash inference.sh
```

## Notes

* Please preprocess data under `data_preprocess/` before training.
* Update all paths in `configs/*.yaml` before use.
* Outputs will be saved to the directory specified in the config.

```