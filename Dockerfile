FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git wget ffmpeg git-lfs software-properties-common

# RUN wget https://developer.download.nvidia.com/compute/cuda/redist/libcublas/linux-x86_64/libcublas-linux-x86_64-11.11.3.6-archive.tar.xz && \
# tar xf libcublas-linux-x86_64-11.11.3.6-archive.tar.xz && \
# mkdir -p /usr/local/cuda/lib64 && \
# mv libcublas-linux-x86_64-11.11.3.6-archive/lib/libcublas.so.11* libcublas-linux-x86_64-11.11.3.6-archive/lib/libcublasLt.so.11* /usr/local/cuda/lib64/ && \
# rm -rf libcublas-linux-x86_64-11.11.3.6-archive.tar.xz libcublas-linux-x86_64-11.11.3.6-archive

COPY . .

RUN git clone https://huggingface.co/fudan-generative-ai/hallo pretrained_models || true
RUN wget -O pretrained_models/hallo/net.pth https://huggingface.co/fudan-generative-ai/hallo/resolve/main/hallo/net.pth?download=true

RUN python3 -m venv venv && \
    . venv/bin/activate && \ 
    pip install -r requirements.txt && \ 
    pip install -e .

RUN pip install torch==2.2.2+cu121 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install onnxruntime-gpu

EXPOSE 7860

RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]

