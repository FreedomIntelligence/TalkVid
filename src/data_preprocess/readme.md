# Data Processing Scripts

## 1. face-crop

Crop the input face video directory into 512x512 square frames with the face centered. This step can be skipped for the HDTF dataset, as the videos in that dataset already meet the requirement.

```bash
bash scripts/data_process/face_crop.sh 
```

## 2. face-info

Given the cropped face video directory, extract face-related parameters.

```bash
bash scripts/data_process/extract_face_info.sh
```

## 3. audio-emb

Given a directory of videos with audio (HDTF) or a directory of audio files (TalkVid), extract audio features using wav2vec2.

```bash
bash scripts/data_process/extract_audio_emb.sh 
```

# Run

Before running the scripts, the expected dataset directory structure is as follows:

```bash
/path/to/dataset/    # TalkVid or HDTF; HDTF only contains videos/
├── audios/
│   ├── file1.m4a
│   ├── file2.m4a
│   └── ...
├── videos/
│   ├── file1.mp4
│   ├── file2.mp4
│   └── ...
```

To run the scripts:

```bash
cd data_preprocess

# create conda env
bash env.sh

# download necessary hf-model-ckpts
bash download_hf.sh

# 1. Face cropping
bash scripts/data_process/face_crop.sh
# 2. Extract audio features
bash scripts/data_process/extract_audio_emb.sh
# 3. Extract face info
bash scripts/data_process/extract_face_info.sh

## Steps 1 and 2 can be executed in parallel. Step 3 requires the output of Step 1,
## so it can only start after Step 1 is completed (fully or partially).
```

After data processing is complete, the expected dataset directory structure is:

```bash
/path/to/dataset/
├── audios/
│   ├── 0000.m4a
│   ├── 0001.m4a
│   └── ...
├── new_face_info/
│   ├── 0000.pt
│   ├── 0001.pt
│   └── ...
├── short_clip_aud_embeds/
│   ├── 0000.pt
│   ├── 0001.pt
│   └── ...
├── videos/
│   ├── 0000.mp4
│   ├── 0001.mp4
│   └── ...
└── videos-crop/
    ├── 0000.mp4
    ├── 0001.mp4
    └── ...
```

Sanity checks:

```bash
# 1. Check whether the number of frames in face-info and audio-emb matches those in videos-crop and videos
bash scripts/check/check_face_and_audio.sh
# 2. Check whether fps and frame count of videos-crop meet expectations (e.g., fps=24, frames=121)
python scripts/check/check_fps_frames.py
# 3. For finer-grained checks, see scripts/check/check_audio.txt
```

Create the JSON file for training:

```python
python scripts/utils/create_data_json.py
```

Example output JSON format:

```json
[
  {
    "video": "/data/TalkVid/videos-crop/videovideo-0F1owya2oo-scene20_scene2.mp4",
    "face_info": "/data/TalkVid/new_face_info/videovideo-0F1owya2oo-scene20_scene2.pt",
    "audio_embeds": "/data/TalkVid/short_clip_aud_embeds/videovideo-0F1owya2oo-scene20_scene2.pt"
  },
  {
    "video": "/data/TalkVid/videos-crop/videovideo-0F1owya2oo-scene5_scene1.mp4",
    "face_info": "/data/TalkVid/new_face_info/videovideo-0F1owya2oo-scene5_scene1.pt",
    "audio_embeds": "/data/TalkVid/short_clip_aud_embeds/videovideo-0F1owya2oo-scene5_scene1.pt"
  },
  ...
]
```
