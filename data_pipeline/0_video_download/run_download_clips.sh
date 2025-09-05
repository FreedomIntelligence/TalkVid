#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status,
# if an undefined variable is used, or if any command in a pipeline fails.
set -euo pipefail

# -----------------------------------------------------------------------------
# Bash replacement for run_download_clips.ps1
# -----------------------------------------------------------------------------
# Usage:
#   ./run_download_clips.sh [INPUT_JSON] [OUTPUT_DIR] [LIMIT] [WORKERS] [COOKIES] [BROWSER] [EXTRACTOR_ARGS]
#
# Positional arguments (all optional, defaults match the PowerShell script):
#   INPUT_JSON       Path to filtered_video_clips.json (default: script_dir/filtered_video_clips.json)
#   OUTPUT_DIR       Directory where clips will be downloaded (default: script_dir/clips_output)
#   LIMIT            Max number of clips to download; 0 means no limit (default: 10)
#   WORKERS          Number of parallel download workers (default: 4)
#   COOKIES          Cookie option passed to yt-dl (default: --cookies-from-browser)
#   BROWSER          Browser name for cookie extraction (default: firefox)
#   EXTRACTOR_ARGS   Extra extractor args string (default: empty)
# -----------------------------------------------------------------------------

# Resolve the directory in which this script resides
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Read positional parameters or fall back to defaults
INPUT_JSON="${1:-$script_dir/filtered_video_clips.json}"
OUTPUT_DIR="${2:-$script_dir/clips_output}"
LIMIT="${3:-10}"
WORKERS="${4:-4}"
COOKIES="${5:---cookies-from-browser}"
BROWSER="${6:-firefox}"
EXTRACTOR_ARGS="${7:-}"

# Show configuration
echo "Input: $INPUT_JSON"
echo "Output: $OUTPUT_DIR"
if [[ "$LIMIT" -gt 0 ]]; then echo "Limit: $LIMIT"; fi
echo "Workers: $WORKERS"
if [[ -n "$BROWSER" ]]; then echo "Browser cookies: $BROWSER"; fi
if [[ -n "$EXTRACTOR_ARGS" ]]; then echo "Extractor args: $EXTRACTOR_ARGS"; fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Build argument list for download_clips.py
args=(--input "$INPUT_JSON" --output "$OUTPUT_DIR" --workers "$WORKERS")

if [[ "$LIMIT" -gt 0 ]]; then args+=(--limit "$LIMIT"); fi
if [[ -f "$COOKIES" ]]; then args+=(--cookies "$COOKIES"); fi
if [[ -n "$BROWSER" ]]; then args+=(--browser "$BROWSER"); fi
if [[ -n "$EXTRACTOR_ARGS" ]]; then args+=(--extractor_args "$EXTRACTOR_ARGS"); fi

# Execute the Python downloader
python "$script_dir/download_clips.py" "${args[@]}"

exit_code=$?

# Propagate exit code to caller
exit "$exit_code"
