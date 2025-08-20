import argparse
import os, cv2, subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import numpy as np
from insightface.app import FaceAnalysis
from imageio_ffmpeg import get_ffmpeg_exe

## 模型全局加载
APP = None

def init_worker(gpu_id: int, root: str = './model_ckpts/insightface_models/'):
    global APP
    from insightface.app import FaceAnalysis

    APP = FaceAnalysis(
        providers=['CUDAExecutionProvider'],
        provider_options=[{'device_id': 0}],  # 每个进程只看到 1 块卡
        allowed_modules=['detection', 'landmark_3d_68'],
        root=root
    )
    APP.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
    # 关闭 OpenCV 多线程（可选，避免线程争用）
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def _get_crop_from_bbox(center, box_size, width, height, size):
    w = int(box_size / 2)
    left = int(center[0] - w)
    top = int(center[1] - w)
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    return [width, height, left, top, 2 * w, 2 * w]

def process_video(args):
    video_file, input_dir, output_dir, size = args
    
    try:
        video_path = os.path.join(input_dir, video_file)
        output_file = os.path.join(output_dir, video_file)
        if os.path.exists(output_file):
            return f"skip {output_file} since it already exists"

        app = APP
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 由于数据集的视频均为fps=24，无需再进行fps统一处理
        if fps != 24:
            cap.release()
            tmp_path =  video_path.replace(".mp4", "_24.mp4")
            cmd = f'{get_ffmpeg_exe()} -y -i "{video_path}" -r 24 "{tmp_path}" -loglevel quiet'
            subprocess.call(cmd, shell=True)
            os.remove(video_path)
            os.rename(tmp_path, video_path)
            cap = cv2.VideoCapture(video_path)
            fps = 24

        frames = []
        bbox_min = [np.inf, np.inf]
        bbox_max = [-np.inf, -np.inf]

        for _ in range(frame_count):
            still_reading, frame = cap.read()
            if not still_reading:
                break
            preds = app.get(frame)
            if len(preds) == 0:
                continue
            x1, y1, x2, y2 = preds[0].bbox
            bbox_min[0] = min(bbox_min[0], x1)
            bbox_min[1] = min(bbox_min[1], y1)
            bbox_max[0] = max(bbox_max[0], x2)
            bbox_max[1] = max(bbox_max[1], y2)
            frames.append(frame)

        if len(frames) == 0:
            return f"[x] No face detected in {video_file}"

        center_x = int((bbox_min[0] + bbox_max[0]) / 2)
        center_y = int((bbox_min[1] + bbox_max[1]) / 2)
        center = [center_x, center_y]
        box_size = int(max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]) * 1.6)

        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_file, fourcc, fps, (size, size))

        for frame in frames:
            crop_info = _get_crop_from_bbox(center, box_size, frame.shape[1], frame.shape[0], size)
            face_img = frame[crop_info[3]:crop_info[3] + crop_info[5], crop_info[2]:crop_info[2] + crop_info[4]]
            face_img = cv2.resize(face_img, (size, size), interpolation=cv2.INTER_NEAREST)
            writer.write(face_img)

        writer.release()
        # 使用 ffmpeg 转码为 H.264 编码（libx264）
        h264_output_file = output_file.replace(".mp4", "_h264.mp4")
        cmd = f'{get_ffmpeg_exe()} -y -i "{output_file}" -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p "{h264_output_file}" -loglevel error'
        subprocess.call(cmd, shell=True)
        os.remove(output_file)
        os.rename(h264_output_file, output_file)

        return f"[✓] Saved cropped video to: {output_file}"
    except Exception as e:
        return f"Error processing {video_file}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-v", type=str, required=True, help="输入视频所在目录")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="输出视频所在目录")
    parser.add_argument('--model_root', type=str, default='./model_ckpts/insightface_models/')
    parser.add_argument("--size", type=int, default=512, help="裁剪后视频的分辨率")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use for face detection")
    parser.add_argument("--num_workers", "-w", type=int, default=8, help="并行进程数")
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument("--shard", choices=['True', 'true', 'False', 'false'],
                        default='true', help="是否对输入的视频文件夹分片")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ## 二次处理逻辑
    exist_files = set(os.listdir(args.output_dir))
    video_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.mp4')
                          and f not in exist_files])
    
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

    task_args = [(f, args.input_dir, args.output_dir, args.size) for f in shard_files]

    start = time.time()
    with Pool(args.num_workers,
              initializer=init_worker,
              initargs=(args.gpu_id, args.model_root)) as pool:
        for result in tqdm(pool.imap_unordered(process_video, task_args), total=len(task_args)):
            print(result)
    
    end = time.time()
    print(f"\n✅ 所有视频处理完成，共耗时 {((end - start) / 60):.2f} 分钟，进程数为 {args.num_workers}")

if __name__ == "__main__":
    main()
