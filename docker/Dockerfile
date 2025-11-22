FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PATH="/opt/conda/bin:${PATH}" \
	DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 安裝基本工具 (包含 SSH 伺服器)
RUN apt-get update && apt-get install -y \
    openssh-server sudo \
    vim zip net-tools git curl wget tree \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- 1. 安裝 hloc/colmap 需要的「系統依賴」 ----
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
    pillow-heif \
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
RUN /opt/conda/bin/pip install --no-build-isolation -e . 

# ---- 4. SSH 設定 (Root User) ----
# 允許 root 透過 public key 登入，禁止密碼登入
RUN echo "PermitRootLogin prohibit-password" >> /etc/ssh/sshd_config
RUN echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

# 建立 SSH 目錄
RUN mkdir -p /root/.ssh
RUN chmod 700 /root/.ssh
RUN touch /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# 準備 SSH 服務目錄
RUN mkdir -p /run/sshd
RUN chmod 755 /run/sshd

# 將 conda 路徑加入環境變數
RUN echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/profile.d/conda_path.sh

# 預設工作目錄
WORKDIR /workspace

# ---- 5. 啟動腳本 ----
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 22
CMD ["/entrypoint.sh"]