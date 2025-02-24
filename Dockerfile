# Hugging Face 및 PyTorch 환경이 포함된 컨테이너
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    git wget curl vim htop unzip \
    python3 python3-pip python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# Python 환경 및 패키지 설치
RUN pip3 install --upgrade pip
RUN pip3 install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    transformers accelerate huggingface_hub sentencepiece

# NVIDIA 컨테이너 런타임 활성화
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 컨테이너 실행 시 기본 실행 명령어
CMD ["bash"]
