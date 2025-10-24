FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 安裝 SSH 伺服器和基本工具
RUN apt-get update && apt-get install -y \
    openssh-server sudo \
    git curl wget tree \
    python3-pip \
    # --- 在此加入您想提供的所有工具 ---
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- 1. 安裝 hloc/colmap 需要的「系統依賴」 ----
# PyTorch 映像檔沒有 colmap，所以我們必須手動安裝
RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \
    build-essential cmake ninja-build \
    ffmpeg libgl1 libglib2.0-0 libgtk2.0-dev \
    colmap \
    x11-apps \
 && rm -rf /var/lib/apt/lists/*

# ---- 2. 安裝 hloc 需要的「Python 依賴」 ----
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
	--index-url https://download.pytorch.org/whl/cu128 \
	--extra-index-url https://pypi.org/simple \
	jupyterlab \
    xformers \
    transformers \
    huggingface-hub \
    accelerate \
    pycolmap \
	open3d \
    opencv-python \
    kornia \
    matplotlib \
    scipy \
    tqdm \
    h5py \
    pandas \
    einops \
    faiss-cpu \
    safetensors \
    scikit-learn

# ---- 3. 安裝 HLOC ----
WORKDIR /opt
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization hloc
WORKDIR /opt/hloc
RUN pip install --no-build-isolation -e . 

# 預設工作目錄
WORKDIR /workspace



# ---- 3. 安裝 HLOC ----
# 2. 建立一個非 root 的開發者帳號
RUN useradd -m -s /bin/bash developer
# 讓 developer 可以免密碼 sudo
RUN echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 3. 準備 SSH 設定
RUN mkdir -p /home/developer/.ssh
RUN chmod 700 /home/developer/.ssh
# 建立 authorized_keys 檔案，但保持空白
RUN touch /home/developer/.ssh/authorized_keys
RUN chmod 600 /home/developer/.ssh/authorized_keys
RUN chown -R developer:developer /home/developer/.ssh

# 4. 準備一個啟動腳本 (entrypoint)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 暴露 SSH 埠號
EXPOSE 22

# 5. 使用啟動腳本來啟動容器
CMD ["/entrypoint.sh"]