FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0

# 声明接收 devcontainer.json 传来的构建参数
ARG HTTP_PROXY
ARG HTTPS_PROXY

# 将其转化为构建阶段的环境变量 (注意大小写兼容性)
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY

# 避免交互式前端询问
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /workspace

# 安装系统基础依赖 (OpenCV 必需库, git, curl 等)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3.10 \
    && rm -rf /var/lib/apt/lists/*

# 预先安装核心依赖 (利用缓存层)
# 直接指定 python3.10 调用 pip
RUN python3.10 -m pip install --no-cache-dir --default-timeout=1000 \
    "numpy>=1.24" \
    "pandas>=1.3" \
    "opencv-contrib-python==4.10.0.84" \
    "matplotlib" \
    "colorlog" 

# 设置环境变量
# 使用变量扩展语法，防止 PYTHONPATH 未定义时产生前导冒号或报错
ENV PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/workspace/PaddleX"
ENV OMP_NUM_THREADS=2