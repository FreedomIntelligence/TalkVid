import os
import argparse
import torch
import torchvision
import torchaudio
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

STAN_AUD_FPS = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
AUDIO_PROCESSOR = None
AUDIO_ENCODER = None

def init_worker(gpu_id: int, root: str = './model_ckpts/wav2vec2-base-960h/'):
    global AUDIO_PROCESSOR, AUDIO_ENCODER
    if not os.path.isdir(root):
        print(f"模型路径 '{root}' 不存在，改为使用 Hugging Face 默认模型 'facebook/wav2vec2-base-960h'")
        root = "facebook/wav2vec2-base-960h"

    AUDIO_ENCODER = Wav2Vec2Model.from_pretrained(root).to(dtype=torch.float32, device='cuda')
    AUDIO_PROCESSOR = Wav2Vec2Processor.from_pretrained(root)


def prepare_audio_embeddings(audio_waveform, audio_processor, audio_encoder, device=DEVICE, dtype=DTYPE):
    audio_waveform = audio_processor(audio_waveform, return_tensors="pt", sampling_rate=16000)['input_values']
    audio_waveform = audio_waveform.to(device, dtype)
    audio_embeddings = audio_encoder(audio_waveform).last_hidden_state  # [1, num_embeds, d]
    audio_embeddings = audio_embeddings.permute(1, 0, 2)  # [num_embeds, 1, d]
    return audio_embeddings


def process_audio(args):
    video_file, input_dir, audio_dir, output_dir, mode_audio = args

    try:
        video_path = os.path.join(input_dir, video_file)
        original_audio_path = os.path.join(audio_dir, video_file.replace('.mp4', '.m4a'))
        aud_embeds_path = os.path.join(output_dir, video_file.replace('.mp4', '.pt'))
        
        if os.path.exists(aud_embeds_path):
            audio_emb = torch.load(aud_embeds_path)
            if audio_emb is None:
                print(f'Audio embedding at {aud_embeds_path} is None, reprocessing {video_path}.')
            else:
                return f"[✓] Skipped: {video_file}"

        audio_processor = AUDIO_PROCESSOR
        audio_encoder = AUDIO_ENCODER
        
        if mode_audio.lower() == "true":            
            audio_waveform, audio_sampling_rate = librosa.load(original_audio_path, sr=STAN_AUD_FPS, mono=False)
            audio_waveform = torch.from_numpy(audio_waveform)

            ## torchaudio 读取.m4a音频较慢，应使用librosa
            # audio_waveform, audio_sampling_rate = torchaudio.load(original_audio_path)
            # print(f"{audio_waveform.shape}")  # e.g. torch.Size([2, 80992])
        else:
            ## torchaudio 更快，但需安装ffmpeg后端
            audio_waveform, audio_sampling_rate = torchaudio.load(video_path)
            
            ## 如果没有安装ffmpeg后端，则使用下面代码
            # _, audio_waveform, meta_info = torchvision.io.read_video(video_path, pts_unit='sec')
            # if 'audio_fps' not in meta_info:
            #     print(f"❌ 无法读取音频: {video_path}")
            #     return
            # else:
            #     audio_sampling_rate = meta_info['audio_fps']

        # 如果音频采样率不为标准采样率，进行重采样
        if audio_sampling_rate != STAN_AUD_FPS:
            audio_waveform = torchaudio.functional.resample(
                audio_waveform,
                orig_freq=audio_sampling_rate,
                new_freq=STAN_AUD_FPS,
            )

        # 取音频的单通道（去掉多通道）
        audio_waveform = audio_waveform.mean(dim=0)
        
        # 计算音频嵌入
        with torch.no_grad():
            audio_embedding = prepare_audio_embeddings(audio_waveform, audio_processor, audio_encoder, DEVICE, DTYPE)

        # 保存音频嵌入
        if aud_embeds_path is not None:
            torch.save({'global_embeds': audio_embedding}, aud_embeds_path)
            return f"[✓] Saved: {video_file}"
        else:
            return f"[x] Failed: {video_file} (aud_embeds_path is None)"
        
    except Exception as e:
        return f"[x] Failed: {video_file} | {e}"


def main():
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Extract audio embeddings from video files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input video files.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing original audio files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save audio embeddings.")
    parser.add_argument("--model_root", type=str, default='./model_ckpts/wav2vec2-base-960h/', help="Path to the Wav2Vec2 model.")
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument("--shard", choices=['True', 'true', 'False', 'false'],
                        default='true', help="是否对输入的视频文件夹分片")
    parser.add_argument('--mode_audio', choices=['True', 'true', 'False', 'false'],
                        default='false', help="Whether to use original audio files, otherwise extract from video.")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    ## 二次处理逻辑
    exist_files = set(os.listdir(args.output_dir))
    video_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.mp4')
                          and f.replace(".mp4", ".pt") not in exist_files])
    
    print(f"Found {len(video_files)} videos to process in {args.input_dir}.")

    # 每卡处理的视频数量
    if args.shard:
        videos_per_shard = (len(video_files) + args.num_gpus - 1) // args.num_gpus
        start = (args.gpu_id) * videos_per_shard
        end = (args.gpu_id + 1) * videos_per_shard
        shard_files = video_files[start:end]
    else:
        shard_files = video_files

    print(f"GPU {args.gpu_id} | Processing {len(shard_files)} videos...")

    task_args = [(f, args.input_dir, args.audio_dir, args.output_dir, args.mode_audio) for f in shard_files]
    
    with Pool(args.num_workers,
              initializer=init_worker,
              initargs=(args.gpu_id, args.model_root)) as pool:
        for result in tqdm(pool.imap_unordered(process_audio, task_args), total=len(task_args)):
            print(result)

if __name__ == "__main__":
    main()
