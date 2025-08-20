import os
import cv2
import torch
from tqdm import tqdm
from insightface.app import FaceAnalysis
from multiprocessing import Pool
import argparse

DET_SIZE = 640  # 512
APP = None

def init_worker(gpu_id: int, root: str = './model_ckpts/insightface_models/'):
    global APP
    from insightface.app import FaceAnalysis

    APP = FaceAnalysis(
        providers=['CUDAExecutionProvider'],
        provider_options=[{'device_id': 0}], # 每个进程只看到一张卡
        root=root,
    )
    APP.prepare(ctx_id=0, det_thresh=0.5, det_size=(DET_SIZE, DET_SIZE))
    # 关闭 OpenCV 多线程（可选，避免线程争用）
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def process_video(args):
    video_file, input_dir, output_dir = args

    try:
        vid_path = os.path.join(input_dir, video_file)
        face_info_path = os.path.join(output_dir, video_file.replace('.mp4', '.pt'))
        if os.path.exists(face_info_path):
            return f"[✓] Skipped: {video_file}"

        app = APP
        frames = []
        cap = cv2.VideoCapture(vid_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        face_info = []
        for frame in frames:
            faces = app.get(frame)
            if len(faces) != 1:
                return f"[x] Dropped: {video_file} (face count != 1)"
            face_info.append([{
                'bbox': face.bbox,
                'kps': face.kps,
                'det_score': face.det_score,
                'landmark_3d_68': face.landmark_3d_68,
                'pose': face.pose,
                'landmark_2d_106': face.landmark_2d_106,
                'gender': face.gender,
                'age': face.age,
                'embedding': face.embedding,
            } for face in faces])

        torch.save(face_info, face_info_path)
        return f"[✓] Saved: {video_file}"
    except Exception as e:
        return f"[x] Failed: {video_file} | {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_root', type=str, default='./model_ckpts/insightface_models/')
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument("--shard", choices=['True', 'true', 'False', 'false'],
                        default='true', help="是否对输入的视频文件夹分片")
    args = parser.parse_args()

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

    task_args = [(f, args.input_dir, args.output_dir) for f in shard_files]
    with Pool(args.num_workers,
              initializer=init_worker,
              initargs=(args.gpu_id, args.model_root)) as pool:
        for result in tqdm(pool.imap_unordered(process_video, task_args), total=len(task_args)):
            print(result)

if __name__ == "__main__":
    main()
