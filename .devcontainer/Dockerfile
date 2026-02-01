FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0

# 避免交互式前端询问
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /workspace

# 1. 更换 apt 源到阿里云 (加速下载)
# 2. 安装系统基础依赖 (OpenCV 必需库, git, curl 等)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv (极速包管理器)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv


# 预先安装核心依赖 (利用缓存层)
# 这里手动列出一些大包，利用 Docker 层缓存加速后续构建
RUN uv pip install --system \
    "numpy>=1.24" \
    "pandas>=1.3" \
    "opencv-contrib-python==4.10.0.84" \
    "matplotlib" \
    "colorlog" 

# 设置环境变量
# 使用变量扩展语法，防止 PYTHONPATH 未定义时产生前导冒号或报错
ENV PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/workspace/PaddleX"
ENV OMP_NUM_THREADS=2
