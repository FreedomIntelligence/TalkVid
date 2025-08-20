from huggingface_hub import hf_hub_download
import os


os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


repo_id = "facebook/cotracker3"
filename = "scaled_offline.pth"


file_path = hf_hub_download(repo_id=repo_id, filename=filename)

print(f"file is downloaded to: {file_path}")
