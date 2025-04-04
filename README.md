# <img src="assets/emoji.png" alt="emoji" width="35"> ClimateBench-M: A Multi-modal Climate Data Benchmark 

<img src="assets\climatebench-m-overview.png">

|[**🗺️ Climate Time Series Data**](https://huggingface.co/datasets/Violet24K/ClimateBench-M-TS) | [**🛰️ Satellite Image Data**](https://huggingface.co/datasets/Violet24K/ClimateBench-M-IMG) | [**📖 Paper**](https://arxiv.org/abs/2404.00225) |

**ClimateBench-M** is the first **multi-modal climate benchmark** designed to support the development of artificial general intelligence (AGI) in climate applications. It aligns data across three critical modalities at a unified spatio-temporal resolution:

1. Time-series climate variables from ERA5.
2. Extreme weather event records from NOAA.
3. Satellite imagery from NASA HLS.



## 📦 Dataset Download
To facilitate broad accessibility and reproducibility, **ClimateBench-M** is hosted on [🤗 Hugging Face Datasets](https://huggingface.co/docs/datasets/index). Due to its size and multi-modal nature, the dataset is split into two parts:

- [**ClimateBench-M-TS**](https://huggingface.co/datasets/Violet24K/ClimateBench-M-TS): Climate time-series with extreme weather event aligned and labeled.
- [**ClimateBench-M-IMG**](https://huggingface.co/datasets/Violet24K/ClimateBench-M-IMG): Satellite imagery data

This separation allows users to download only the modalities relevant to their tasks. To download, please ensure you have the Hugging Face CLI installed:

```sh
huggingface-cli login
python scripts/dataset_download.py
```

The data is downloaded into Data/ folder by default. If you change the path to download data, please also change the data loading path for downstream tasks.


## 🌤️ Weather Forecasting
### 🛠 Environment Setup
We provide a pre-built Docker image for ease of use. With Docker installed,
```sh
docker pull violet24k/climatebench-m-ts:latest
docker run --gpus all -it -v .:/workspace -w /workspace violet24k/climatebench-m-ts:latest bash
```
Alternatively, you can set up the environment manually by
```sh
conda create -n climatebench-m-ts python=3.11.11
conda activate climatebench-m-ts
# install torch, example:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# install other dependencies
pip install pandas geopandas scikit-learn pyarrow rasterio matplotlib huggingface_hub
```

### 🧹 Data Preprocessing
The previous download gives the raw data for more flexible use case. For the weather forecasting task, you can download our processed data from the downloaded raw data ***or*** process the data by yourself.
```sh
# download our processed data
python scripts/dataset_download_ts_processed.py
# OR process the data by yourself
python Task/weather_forecasting/weather_data_processing.py
```

### 🔮 Our Generative Model For Weather Forecasting
Then, you can create your own model under Task/weather_forecasting/YOUR_MODEL_NAME. We provide a generative model baseline SGM_Time_Series for reference. Our generative model first discovers temporally fine-grained causal relationships from the processed data.
```sh
# Might take several hours to execute
python Task/weather_forecasting/SGM_Time_Series/finding_causality.py
# OR download our trained causality files
python scripts/dataset_download_ts_savedmodules.py
```
After the causality (0_K_time_conv.pt, 0_K_feature_encoder.pt, 0_K_feature_decoder.pt, 0_K_best_ELBO_graph_seq.npy) is stored in saved_modules folder, one can train and evaluate our SGM model by
```sh
python Task/weather_forecasting/SGM_Time_Series/forecasting.py
python Task/weather_forecasting/SGM_Time_Series/evaluation.py
```

## ⛈️ Anomaly Detection
The anomaly detection task uses the same climate time series data as the weather forecasting task, with additional anomaly labels included. When you run Task/weather_forecasting/SGM_Time_Series/evaluation.py, it will evaluate performance on both weather forecasting and anomaly detection tasks.


## 🌾 Crop Segmentation
### 🛠 Environment Setup
For image tasks, following the [NASA IMPACT repo](https://github.com/NASA-IMPACT/hls-foundation-os/), we need openmim, mmcv, and mmsegmentation.
We also provide a pre-built Docker image for ease of use:
```sh
docker pull violet24k/climatebench-m-img:latest
docker run --gpus all -it -v .:/workspace -w /workspace violet24k/climatebench-m-img:latest bash
```
Alternatively, to set up the environment manually,
```sh
cd Task/crop_segmentation/SGM_Image
conda create -n climatebench-m-img python==3.9
conda activate climatebench-m-img
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115
pip install -e .
pip install -U openmim
mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html
pip install numpy==1.24.0
```
For more details of the installed packages, please refer to [PyTorch](https://pytorch.org/get-started/locally/), [openmim](https://openmim.readthedocs.io/en/latest/installation.html), [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html), and [MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/). Note: The original NASA IMPACT repo uses mmcv 1.x and mmsegmentation 0.x. While these are older versions, upgrading to newer releases like mmcv 2.x requires significant changes due to the introduction of [MMEngine](https://mmengine.readthedocs.io/en/latest/).

### Data Split
ClimateBench-M-IMG provides raw satellite image data by default for maximal flexibility. To generate train/validation splits and prepares inputs for model,
```sh
# in working directory Task/crop_segmentation/SGM_Image
python image_data_processing.py
```
To improve performance, we recommend initializing the MAE-based generative model with pretrained weights from IBM’s Prithvi Foundation Model with
```sh
git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
```
⚠️ Important: After cloning, check that the .pt files are ~400MB. If they are tiny, they may be Git LFS pointers — in that case, install Git LFS and run git lfs pull.


### ⚙️ Training & Evaluation
Before running, please double-check Task/crop_segmentation/SGM_Image/configs/multi_temporal_crop_classification.py, especially lines 15, 49, 59, as mim may not properly resolve os.path.

To train and evaluate the MAE-backboned generative model,
```sh
# Train the model
mim train mmsegmentation configs/multi_temporal_crop_classification.py

# Evaluate the model (replace with actual checkpoint path)
mim test mmsegmentation configs/multi_temporal_crop_classification.py \
    --checkpoint path_to_checkpoint_model.pth --eval "mIoU"
```

## Cite
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{DBLP:conf/kdd/ZhengJLTH24,
  author       = {Lecheng Zheng and
                  Baoyu Jing and
                  Zihao Li and
                  Hanghang Tong and
                  Jingrui He},
  editor       = {Ricardo Baeza{-}Yates and
                  Francesco Bonchi},
  title        = {Heterogeneous Contrastive Learning for Foundation Models and Beyond},
  booktitle    = {Proceedings of the 30th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2024, Barcelona, Spain, August 25-29, 2024},
  pages        = {6666--6676},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3637528.3671454},
  doi          = {10.1145/3637528.3671454},
  timestamp    = {Sun, 08 Sep 2024 16:05:58 +0200},
  biburl       = {https://dblp.org/rec/conf/kdd/ZhengJLTH24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```