import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple
import json
import math
import glob 
from pathlib import Path 
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)  # type: ignore




def get_video_id(url: str) -> str:
    """从 YouTube URL 中提取视频 ID。"""
    if 'watch?v=' in url:
        return url.split('watch?v=')[-1].split('&')[0]
    if 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    if '/shorts/' in url:
        return url.split('/shorts/')[-1].split('?')[0]
    # Fallback for other URL formats or just return a hash
    # For simplicity, we'll just use the last part of the URL
    return url.split('/')[-1] or "unknown_id"


def find_executable(candidates: List[str]) -> Optional[str]:
    """Return the first existing candidate executable path or None."""
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
        # Also consider a relative executable within the current directory
        if os.path.isfile(candidate):
            return candidate
    return None


def safe_mkdir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def is_clip_downloaded(
    url: str, start: float, end: float, output_dir: str
) -> bool:
    """通过检查 JSON 日志来判断片段是否已成功下载。"""
    video_id = get_video_id(url)
    # 规范化文件名中的浮点数格式
    log_filename = f"{video_id}_{start:.3f}_{end:.3f}.json".replace(":", "-")
    log_file = os.path.join(output_dir, "json_logs", log_filename)

    if not os.path.exists(log_file):
        return False

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        if log_data.get("download_info", {}).get("status") == "success":
            # 额外检查视频文件是否真实存在
            video_file = log_data.get("download_info", {}).get("video_clip_file")
            if video_file and os.path.exists(video_file) and os.path.getsize(video_file) > 0:
                return True
    except (json.JSONDecodeError, KeyError):
        return False
    return False


def iter_segments_from_big_json(
    input_json_path: str,
) -> Generator[Tuple[str, float, float], None, None]:
    """
    直接加载并遍历 JSON 文件。

    每一项（dict）应包含键：
        - "Video Link"（或小写变体）
        - "start-time" / "start"
        - "end-time" / "end"

    返回 (url, start, end) 元组。
    """

    with open(input_json_path, "r", encoding="utf-8") as f:
        try:
            items = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法解析 JSON 文件: {input_json_path} | {exc}") from exc

    if not isinstance(items, list):
        raise ValueError("期望 JSON 顶层为数组（list）")

    for item in items:
        if not isinstance(item, dict):
            continue

        info_dict = item.get("info", {})
        url = info_dict.get("Video Link") or info_dict.get("video_link")
        start_val = item.get("start-time") or item.get("start")
        end_val = item.get("end-time") or item.get("end")

        if url is None or start_val is None or end_val is None:
            continue

        try:
            start_f = float(start_val)
            end_f = float(end_val)
        except (TypeError, ValueError):
            continue

        if end_f > start_f:
            yield (url, start_f, end_f)


def seconds_to_time_string(seconds_value: float) -> str:
    if seconds_value < 0:
        seconds_value = 0.0
    ms = math.floor(seconds_value * 1000 + 1e-6)  # 向下取整到毫秒
    hours, rem = divmod(ms, 3600_000)
    minutes, ms = divmod(rem, 60_000)
    seconds, ms = divmod(ms, 1000)
    if ms == 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

def get_yt_dlp_base_cmd(cookies_path: Optional[str], browser: Optional[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """Build base yt-dlp command, preferring python -m yt_dlp. Returns (cmd, error)."""
    try:
        import importlib.util  # noqa: F401

        if importlib.util.find_spec("yt_dlp") is not None:
            base_cmd = [sys.executable, "-m", "yt_dlp"]
        else:
            raise ImportError
    except Exception:
        yt_dlp_candidates = [
            os.path.join(os.getcwd(), "yt-dlp_x86.exe"),
            os.path.join(os.getcwd(), "yt-dlp.exe"),
            "yt-dlp",
        ]
        yt_dlp_path = find_executable(yt_dlp_candidates)
        if yt_dlp_path is None:
            return None, "yt-dlp not found. Install with: python -m pip install --user -U yt-dlp"
        base_cmd = [yt_dlp_path]

    # Cookies are added by callers depending on the operation (probe/download)
    if browser:
        base_cmd = [*base_cmd, "--cookies-from-browser", browser]
    elif cookies_path and os.path.exists(cookies_path):
        base_cmd = [*base_cmd, "--cookies", cookies_path]

    # Use IPv4 and ignore user config files for reproducibility
    base_cmd = [*base_cmd, "-4", "--ignore-config"]
    return base_cmd, None

def run_yt_dlp_multi_sections(
    url: str,
    segments: List[Tuple[float, float]],
    output_dir: str,
    cookies_path: Optional[str] = None,
    browser: Optional[str] = None,
    extractor_args: Optional[str] = None,
    strict_cuts: bool = False,   # True = 更准的切口（会重编码，慢）；False = 更快（关键帧附近）
) -> Tuple[int, str]:
    """
    对同一 URL 的多个片段，合并为一次 yt-dlp 调用（多个 --download-sections）。
    产物文件名使用 section 变量，避免覆盖。
    """
    safe_mkdir(output_dir)
    base_cmd, err = get_yt_dlp_base_cmd(cookies_path, browser)
    if base_cmd is None:
        return 1, err or "Unable to locate yt-dlp"

    # --- 构造下载命令 ---
    video_id = get_video_id(url)
    video_output_dir = os.path.join(output_dir, video_id)
    safe_mkdir(video_output_dir)

    # 构造多段 sections
    section_args: List[str] = []
    for (s, e) in segments:
        if e <= s:
            continue
        s_str = seconds_to_time_string(s)
        e_str = seconds_to_time_string(e)
        section_args.extend(["--download-sections", f"*{s_str}-{e_str}"])

    if not section_args:
        return 1, "No valid segments for this URL"

    # 输出模板：所有文件都放入以 video_id 命名的子目录中
    output_template = os.path.join(
        video_output_dir,
        "%(id)s_%(section_number)03d_%(section_start).3f_%(section_end).3f.%(ext)s",
    )

    cmd: List[str] = [
        *base_cmd,
        "-4",
        "--ignore-config",
        "--no-playlist",
        "--retries", "10",
        "--fragment-retries", "10",
        "--concurrent-fragments", "8",
        "-N", "4",
        "--no-warnings",
        "--restrict-filenames",
        "--no-continue", "--no-overwrites",
        # --- 新增功能 ---
        "--print", "after_move:filepath", # 打印最终文件路径
        "--write-subs",
        "--write-auto-subs",
        "--write-description",
        "--extract-audio",
        "--audio-format", "m4a", "--audio-quality", "0",
        "--keep-video",
        "--no-keep-fragments",  # 不保留中间文件
        "--clean-info-json",  # 清理信息文件
        # --- 输出模板 ---
        "-o", output_template,
        # 尽量拿到 H.264+AAC，可无损 remux；退化到 best 也能跑
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4", 
    ]
    if strict_cuts:
        cmd.append("--force-keyframes-at-cuts")

    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])

    # 拼上多段
    cmd.extend(section_args)
    cmd.append(url)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if proc.returncode == 0:
            return 0, proc.stdout.strip()
        # 简单回退：遇到“格式不可用”就退到 best
        err_msg = (proc.stderr.strip() or proc.stdout.strip())
        if "Requested format is not available" in err_msg:
            fallback_cmd = [
                *base_cmd,
                "-4", "--ignore-config", "--no-playlist",
                "--retries", "10", "--fragment-retries", "10",
                "--concurrent-fragments", "8", "-N", "4",
                "--no-warnings", "--restrict-filenames",
                "-c", "--no-overwrites",
                # --- 新增功能 (回退) ---
                "--print", "after_move:filepath",
                "--write-subs", "--write-auto-subs", "--write-description",
                "--extract-audio", "--audio-format", "m4a", "--keep-video",
                # --- 输出模板 (回退) ---
                "-o", output_template,
                "-f", "bestvideo[ext=mp4][vcodec!=none]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!=none]",
                "--remux-video", "mp4",
            ]
            if strict_cuts:
                fallback_cmd.append("--force-keyframes-at-cuts")
            if extractor_args:
                fallback_cmd.extend(["--extractor-args", extractor_args])
            fallback_cmd.extend(section_args)
            fallback_cmd.append(url)

            proc2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding='utf-8')
            if proc2.returncode == 0:
                return 0, proc2.stdout.strip()
            return proc2.returncode, (proc2.stderr.strip() or proc2.stdout.strip())
        return proc.returncode, err_msg
    except Exception as exc:  # noqa: BLE001
        return 1, f"yt-dlp failed: {exc}"


def probe_url_availability(
    url: str,
    cookies_path: Optional[str],
    browser: Optional[str],
    extractor_args: Optional[str] = None,
) -> Tuple[bool, str]:
    """Check if a URL is available by asking yt-dlp to print the id without downloading."""
    base_cmd, err = get_yt_dlp_base_cmd(cookies_path, browser)
    if base_cmd is None:
        return False, err or "yt-dlp not found"

    cmd: List[str] = [
        *base_cmd,
        "-s",
        "--no-warnings",
        "-O",
        "%(id)s",
        url,
    ]
    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return True, proc.stdout.strip()
        # Collect an error message
        msg = (proc.stderr or proc.stdout or "unknown error").strip()
        return False, msg
    except Exception as exc:  # noqa: BLE001
        return False, f"probe failed: {exc}"


# ---- 解析下载产物 ------------------------------------------------------------


def _match_segment_from_name(name: str) -> Optional[Tuple[float, float]]:
    """从文件名中提取 segment (start,end)。返回 None 如果无法匹配。"""
    m = re.search(r"_(\d+\.\d+)_(\d+\.\d+)\.", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_ytdlp_output(
    output: str,
    segments: List[Tuple[float, float]],
    video_id: str,
    video_output_dir: str,
) -> Dict[Tuple[float, float], Dict]:
    """
    解析 yt-dlp 的输出，将文件路径与原始分段关联。
    现在不仅解析 stdout，还会回退到扫描输出目录，以确保拿到完整的
    audio / description / subtitle 信息。
    """

    # 起始：先从 stdout 粗略提取
    files_from_stdout = [line.strip() for line in output.splitlines() if line.strip()]

    # 分类容器
    description_file: str = ""
    subtitle_files: List[str] = []
    clip_files: Dict[Tuple[float, float], List[str]] = defaultdict(list)

    for raw in files_from_stdout:
        # 截掉前缀标记，如 "[info] Writing video description to: "
        possible_path = raw.split(": ")[-1].strip()
        path = Path(possible_path)
        if not path.exists():
            # 如果提取出来的不是一个真实路径，则跳过
            continue

        if path.name.endswith(".description"):
            description_file = str(path)
        elif path.suffix.lower() in {".vtt", ".srt", ".ass"}:
            subtitle_files.append(str(path))
        elif path.suffix.lower() in {".mp4", ".m4a", ".webm", ".mkv"}:
            seg_match = _match_segment_from_name(path.name)
            if seg_match:
                closest_seg = min(
                    segments,
                    key=lambda s: abs(s[0]-seg_match[0])+abs(s[1]-seg_match[1])
                )
                clip_files[closest_seg].append(str(path))

    # --- 回退方案：扫描输出目录，填补缺失信息 --------------------------------
    try:
        for file_name in os.listdir(video_output_dir):
            file_path = os.path.join(video_output_dir, file_name)
            path = Path(file_path)
            if path.name.endswith(".description") and not description_file:
                description_file = file_path
            elif path.suffix.lower() in {".vtt", ".srt", ".ass"} and file_path not in subtitle_files:
                subtitle_files.append(file_path)
            elif path.suffix.lower() in {".mp4", ".m4a", ".webm", ".mkv"}:
                seg_match = _match_segment_from_name(path.name)
                if seg_match:
                    closest_seg = min(
                        segments,
                        key=lambda s: abs(s[0]-seg_match[0])+abs(s[1]-seg_match[1])
                    )
                    if file_path not in clip_files[closest_seg]:
                        clip_files[closest_seg].append(file_path)
    except FileNotFoundError:
        pass

    # 组装最终结果
    results: Dict[Tuple[float, float], Dict] = {}
    for seg in segments:
        file_list = clip_files.get(seg, [])
        video_file = next((f for f in file_list if f.endswith((".mp4", ".mkv", ".webm"))), "")
        audio_file = next((f for f in file_list if f.endswith(".m4a")), "")

        results[seg] = {
            "video_clip_file": video_file,
            "audio_clip_file": audio_file,
            "description_file": description_file,
            "subtitle_files": subtitle_files,
        }

    return results


def load_unavailable_urls(log_file_path: str) -> set:
    """从日志文件中加载存在永久性错误（如视频不可用）的 URL 集合。"""
    unavailable_urls = set()
    if not os.path.exists(log_file_path):
        return unavailable_urls

    # 常见的永久性错误信息片段（小写）
    permanent_error_phrases = [
        "video unavailable",
        "account associated with this video has been terminated",
        "private video",
        "video is private",
        "user has closed their youtube account",
    ]

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    url, reason = parts[0], parts[1].lower()
                    if any(phrase in reason for phrase in permanent_error_phrases):
                        unavailable_urls.add(url)
    except Exception as e:
        print(f"Warning: Could not read or parse unavailable URLs log: {e}")

    return unavailable_urls


def download_with_ytdlp(
    input_json_path: str,
    output_dir: str,
    cookies_path: Optional[str],
    browser: Optional[str],
    extractor_args: Optional[str],
    limit: Optional[int],
    workers: int,
) -> None:
    """
    改造：同一 URL 的多个片段合并一次 yt-dlp 调用（每个 URL 提交一个任务）。
    仍然做一次可用性探测（每个 URL 一次），失败的 URL 全部跳过。
    新增：基于 JSON 日志的断点续传、详细日志记录、元数据和音轨下载。
    """

    # 1) 先流式解析 + 聚合到内存（url -> list[(start,end)])
    # 如需避免占内存，可把此处换成“按 URL 写临时文件，二次读取”的两阶段策略。
    url2segments: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    count = 0
    total_segments = 0
    skipped_due_to_log = 0

    # 准备日志目录
    json_logs_dir = os.path.join(output_dir, "json_logs")
    safe_mkdir(json_logs_dir)

    for (url, start, end) in iter_segments_from_big_json(input_json_path):
        total_segments += 1
        if limit is not None and limit >= 0 and count >= limit:
            break

        # 基于日志的断点续传检查
        if is_clip_downloaded(url, start, end, output_dir):
            skipped_due_to_log += 1
            continue

        if end > start:
            url2segments[url].append((start, end))
            count += 1
    
    print(f"Total segments found: {total_segments}")
    print(f"Skipped (already downloaded): {skipped_due_to_log}")
    print(f"Segments to download: {count}")


    if not url2segments:
        print("No new segments to download.")
        return

    safe_mkdir(output_dir)
    logs_dir = os.path.join(output_dir, "logs")
    safe_mkdir(logs_dir)
    failed_urls_file = os.path.join(logs_dir, "failed_urls.txt")
    failed_segments_file = os.path.join(logs_dir, "failed_segments.txt")

    # 新增：加载之前已确认不可用的 URL
    unavailable_urls = load_unavailable_urls(failed_urls_file)
    if unavailable_urls:
        print(f"Loaded {len(unavailable_urls)} permanently unavailable URLs from logs.")

    # --- 初始化进度条 ---------------------------------------------------------
    # 使用 rich Progress（或降级实现）
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    progress.__enter__()  # 手动进入，使其可跨多个 with 范围外使用
    task_urls = progress.add_task("URLs", total=len(url2segments))
    task_segments = progress.add_task("Segments", total=count)

    # 2) 逐 URL 探测可用性，构造任务
    tasks = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures_map = {}
        for url, segs in url2segments.items():
            # 新增：跳过已知的不可用 URL，避免重复探测
            if url in unavailable_urls:
                reason = "Skipping probe: URL previously marked as permanently unavailable."
                with open(failed_segments_file, "a", encoding="utf-8") as fseg:
                    for (s, e) in segs:
                        fseg.write(f"SKIP\t{url}\t{s:.3f}\t{e:.3f}\t{reason}\n")
                # 进度条同样推进
                progress.update(task_urls, advance=1)
                progress.update(task_segments, advance=len(segs))
                continue

            ok, reason = probe_url_availability(url, cookies_path, browser, extractor_args)
            if not ok:
                with open(failed_urls_file, "a", encoding="utf-8") as furl:
                    furl.write(f"{url}\t{reason}\n")
                # 也把所有该 URL 的片段记到 failed_segments
                for (s, e) in segs:
                    with open(failed_segments_file, "a", encoding="utf-8") as fseg:
                        fseg.write(f"SKIP\t{url}\t{s:.3f}\t{e:.3f}\t{reason}\n")
                # 同样推进进度条
                progress.update(task_urls, advance=1)
                progress.update(task_segments, advance=len(segs))
                continue

            # （可选）对 segs 做小幅清理：去重 + 排序 + 合并微小重叠/相邻
            segs = sorted(set(segs))
            # 这里不默认合并，以“多文件输出”为准；如需合并相邻（gap<0.2s），放开下段：
            # segs = coalesce_segments(segs, merge_gap=0.2)

            future = executor.submit(
                run_yt_dlp_multi_sections,
                url, segs, output_dir, cookies_path, browser, extractor_args, True
            )
            futures_map[future] = (url, segs)

        # 3) 回收
        for fut in as_completed(futures_map):
            url0, segs0 = futures_map[fut]
            # 更新进度条：无论成功失败都算处理完成
            progress.update(task_urls, advance=1)
            progress.update(task_segments, advance=len(segs0))
            video_id = get_video_id(url0)
            try:
                rc, msg = fut.result()
            except Exception as exc:
                rc, msg = 1, f"Task failed with exception: {exc}"

            if rc != 0:
                print(f"[yt-dlp] Failed: {url0} | {msg}")
                with open(failed_segments_file, "a", encoding="utf-8") as fseg:
                    for (s,e) in segs0:
                        fseg.write(f"FAIL\t{url0}\t{s:.3f}\t{e:.3f}\t{msg}\n")
                continue
            
            # 解析成功下载的文件
            parsed_results = parse_ytdlp_output(msg, segs0, video_id, os.path.join(output_dir, video_id))

            for seg_tuple, files_info in parsed_results.items():
                start, end = seg_tuple
                log_filename = f"{video_id}_{start:.3f}_{end:.3f}.json".replace(":", "-")
                log_file = os.path.join(json_logs_dir, log_filename)

                log_data = {
                    "source_info": {
                        "url": url0,
                        "start_time": start,
                        "end_time": end,
                    },
                    "download_info": {
                        **files_info,
                        "status": "success",
                        "error": "",
                        "download_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                }
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)

            # 检查哪些分段没有成功产物
            succeeded_segs = set(parsed_results.keys())
            failed_segs = [s for s in segs0 if s not in succeeded_segs]
            if failed_segs:
                with open(failed_segments_file, "a", encoding="utf-8") as fseg:
                    for (s,e) in failed_segs:
                        fseg.write(f"FAIL\t{url0}\t{s:.3f}\t{e:.3f}\tNo output file generated\n")

        # 关闭进度条
        progress.__exit__(None, None, None)


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg 未找到，安装后再试（yt-dlp 的分段裁切需要 ffmpeg）。")


def cleanup_final_files(output_dir: str) -> None:
    """根据所有 json_logs 清理未被记录的文件，保持输出目录整洁。"""
    print("\nStarting final file cleanup...")
    json_logs_dir = os.path.join(output_dir, "json_logs")
    if not os.path.isdir(json_logs_dir):
        print("json_logs directory not found, skipping cleanup.")
        return

    # 1. 收集所有记录在案的文件路径
    recorded_files = set()
    json_files = glob.glob(os.path.join(json_logs_dir, "*.json"))
    for log_file in json_files:
        recorded_files.add(os.path.abspath(log_file)) # 把日志本身也加入白名单
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            info = data.get("download_info", {})
            if info.get("status") != "success":
                continue

            for key, value in info.items():
                if key.endswith("_file") and isinstance(value, str) and value:
                    recorded_files.add(os.path.abspath(value))
                elif key.endswith("_files") and isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item:
                            recorded_files.add(os.path.abspath(item))
        except Exception as e:
            print(f"Error processing log file {log_file}: {e}")

    if not recorded_files:
        print("No recorded files found in logs, skipping cleanup.")
        return

    # 2. 遍历输出目录，删除未记录的文件
    print(f"Found {len(recorded_files)} files recorded in logs. Scanning for unrecorded files...")
    deleted_count = 0
    for root, _, files in os.walk(output_dir):
        for file in files:
            # 跳过原始失败日志
            if "logs" in root and ("failed_urls.txt" in file or "failed_segments.txt" in file):
                continue

            file_path = os.path.abspath(os.path.join(root, file))
            if file_path not in recorded_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted unrecorded file: {file_path}")
                except OSError as e:
                    print(f"Failed to delete {file_path}: {e}")
    print(f"Cleanup complete. Deleted {deleted_count} unrecorded files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download clips specified in a large JSON file directly with yt-dlp sections on Windows."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.getcwd(), "filtered_video_clips.json"),
        help="Path to the large JSON file containing 'Video Link', 'start-time', 'end-time' fields.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.getcwd(), "clips_output"),
        help="Directory to store the downloaded clips.",
    )
    parser.add_argument(
        "--mode",
        choices=["ytdlp"],
        default="ytdlp",
        help="仅支持 'ytdlp'，已移除 video2dataset 支持。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many clip segments (for testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent workers for yt-dlp mode.",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=os.path.join(os.getcwd(), "cookies.txt"),
        help="Path to cookies.txt to pass to yt-dlp if present.",
    )
    parser.add_argument(
        "--browser",
        type=str,
        choices=["edge", "chrome", "firefox", "chromium", "brave", "vivaldi", "opera"],
        default=None,
        help="Use --cookies-from-browser <browser> for YouTube auth (recommended).",
    )
    parser.add_argument(
        "--extractor_args",
        type=str,
        default=None,
        help="Pass through to yt-dlp --extractor-args, e.g. 'youtube:player_client=android'",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Run cleanup process after downloading to remove unlogged files.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_ffmpeg()
    args = parse_args()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start downloading with yt-dlp")

    download_with_ytdlp(
        input_json_path=args.input,
        output_dir=args.output,
        cookies_path=args.cookies if os.path.exists(args.cookies) else None,
        browser=args.browser,
        extractor_args=args.extractor_args,
        limit=args.limit,
        workers=args.workers,
    )
    print("yt-dlp clip downloads completed.")

    if args.cleanup:
        cleanup_final_files(args.output)


if __name__ == "__main__":
    main()


