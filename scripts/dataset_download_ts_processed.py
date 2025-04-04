from huggingface_hub import snapshot_download
import os

# get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

save_dir = os.path.join(parent_dir, "Task", "weather_forecasting", "processed_weather_data")


local_dir = snapshot_download(
    repo_id="Violet24K/ClimateBench-M-TS-processed",
    repo_type="dataset",
    local_dir=save_dir,
    local_dir_use_symlinks=False
)