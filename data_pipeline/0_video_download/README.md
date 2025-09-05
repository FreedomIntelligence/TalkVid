# YouTube Video Clips Downloader

A Python-based tool for downloading video clips from YouTube using yt-dlp with support for batch processing and resume functionality.

## Prerequisites

- Python 3.10
- yt-dlp (`pip install yt-dlp`)
- ffmpeg (required for video processing)

## Quick Start

### 1. Prepare Input File

Create a JSON file with video clips data, or download the prepared data from [HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/TalkVid/tree/main/data):

```json
[
  {
    "info": {
      "Video Link": "https://www.youtube.com/watch?v=VIDEO_ID"
    },
    "start-time": 10.5,
    "end-time": 25.3
  }
]
```

### 2. Run the Downloader

**Windows (PowerShell):**
```powershell
.\run_download_clips.ps1 -InputJson "input.json" -OutputDir "output" -Limit 50
```

**Linux/macOS (Bash):**
```bash
./run_download_clips.sh input.json output 50 4 --cookies-from-browser firefox
```

**Direct Python:**
```bash
python download_clips.py --input input.json --output output --limit 50 --workers 4
```

## Command Line Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Input JSON file path | `filtered_video_clips.json` |
| `--output` | Output directory | `clips_output` |
| `--limit` | Maximum clips to download (0 = no limit) | None |
| `--workers` | Number of parallel workers | 4 |
| `--browser` | Browser for cookie extraction | None |
| `--cookies` | Path to cookies.txt file | `cookies.txt` |
| `--extractor_args` | Additional yt-dlp arguments | None |
| `--cleanup` | Remove unlogged files after download | False |

## Features

- **Batch Processing**: Download multiple clips from different videos in parallel
- **Resume Support**: Automatically skip already downloaded clips
- **Cookie Support**: Use browser cookies or cookies.txt for authentication
- **Rich Progress Display**: Real-time progress bars for URLs and segments
- **Comprehensive Logging**: JSON logs for each clip with metadata
- **Audio Extraction**: Downloads both video and audio tracks
- **Subtitle Support**: Downloads available subtitles and descriptions
- **Error Handling**: Robust error handling with detailed failure logs

## Output Structure

```
clips_output/
├── VIDEO_ID/
│   ├── VIDEO_ID_001_10.500_25.300.mp4    # Video clip
│   ├── VIDEO_ID_001_10.500_25.300.m4a    # Audio track
│   └── VIDEO_ID.description               # Video description
├── json_logs/
│   └── VIDEO_ID_10.500_25.300.json       # Download metadata
└── logs/
    ├── failed_urls.txt                    # Failed URL list
    └── failed_segments.txt                # Failed segment details
```

## Authentication

For private or age-restricted videos, use browser cookies:

```bash
# Use browser cookies (recommended)
python download_clips.py --browser firefox --input clips.json

# Or use cookies.txt file
python download_clips.py --cookies cookies.txt --input clips.json
```

## Examples

**Download 100 clips with 8 workers:**
```bash
python download_clips.py --input clips.json --limit 100 --workers 8
```

**Use Chrome cookies and custom extractor args:**
```bash
python download_clips.py --browser chrome --extractor_args "youtube:player_client=android"
```

**Clean up unlogged files after download:**
```bash
python download_clips.py --input clips.json --cleanup
```

## Troubleshooting

1. **ffmpeg not found**: Install ffmpeg and ensure it's in your PATH
2. **yt-dlp not found**: Install with `pip install yt-dlp`
3. **Authentication errors**: Use `--browser` option to extract cookies from your browser
4. **Slow downloads**: Increase `--workers` count or check your network connection

## License

This project is provided as-is for educational and research purposes.