FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y git

WORKDIR /app/SGM_Image
COPY SGM_Image /app/SGM_Image

RUN pip install -e .
RUN pip install -U openmim
RUN mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html
RUN pip install numpy==1.24.0