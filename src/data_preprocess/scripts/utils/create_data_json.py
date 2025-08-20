import os
import json
from tqdm import tqdm
import os

# ==== 可配置参数 ====
video_dir = "/data/TalkVid/videos-crop"                 # 视频目录
face_info_dir = "/data/TalkVid/new_face_info"           # 面部信息目录
audio_embed_dir = "/data/TalkVid/short_clip_aud_embeds" # 音频嵌入目录
output_json_path = "./data/TalkVid.json"                # 输出 JSON 文件路径

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
data = []

for fname in tqdm(os.listdir(face_info_dir), desc="Building JSON"):
    if not fname.endswith(".pt"):
        continue

    base_name = fname.replace(".pt", "")
    video_path = os.path.join(video_dir, base_name + ".mp4")
    face_info_path = os.path.join(face_info_dir, base_name + ".pt")
    audio_embeddings_path = os.path.join(audio_embed_dir, base_name + ".pt")

    # 检查三个文件是否都存在
    if os.path.exists(video_path) and os.path.exists(audio_embeddings_path):
        video_info = {
            "video": video_path,
            "face_info": face_info_path,
            "audio_embeds": audio_embeddings_path
        }
        data.append(video_info)

# 保存为 JSON
with open(output_json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n✅ 完成，共写入 {len(data)} 条记录到 {output_json_path}")
