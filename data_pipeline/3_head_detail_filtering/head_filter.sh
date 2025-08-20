# Get current path
CURRENT_PATH=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"

# Defining config file paths
CONFIG_PATH="${CURRENT_PATH}/../config/config_0_0.yaml"

python head_filter_new.py --config ${CONFIG_PATH}
python src/merge_tmp_jsonls.py --config ${CONFIG_PATH}
python src/filter_out_head_videos.py --config ${CONFIG_PATH}