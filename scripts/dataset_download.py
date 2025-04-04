from huggingface_hub import snapshot_download
import os

# get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

ts_save_dir = os.path.join(parent_dir, "Data", "ClimateBench-M-TS")


local_dir = snapshot_download(
    repo_id="Violet24K/ClimateBench-M-TS",
    repo_type="dataset",
    local_dir=ts_save_dir,
    local_dir_use_symlinks=False
)


img_save_dir = os.path.join(parent_dir, "Data", "ClimateBench-M-IMG")


local_dir = snapshot_download(
    repo_id="Violet24K/ClimateBench-M-IMG",
    repo_type="dataset",
    local_dir=img_save_dir,
    local_dir_use_symlinks=False
)