# =============================================================================
# diffsynth_genwarp 推理镜像
# Python 3.10 + CUDA 12.1 + torchvision RAFT 光流
#
# 构建（在项目根目录执行）：
#   docker build -f docker/Dockerfile -t diffsynth-genwarp:latest .
#
# 运行：
#   docker run --gpus all --rm -it \
#     --shm-size=64G \
#     -v /juicefs-algorithm:/juicefs-algorithm:rw \
#     -v /home/zhuochu_yang:/home/zhuochu_yang \
#     -v /etc/passwd:/etc/passwd:ro \
#     -v /etc/group:/etc/group:ro \
#     --user $(id -u):$(id -g) \
#     -w /juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp \
#     diffsynth-genwarp:latest \
#     bash scripts/flow_warp_sample.sh
#
# 注意：模型权重不打包进镜像，运行时通过 -v /juicefs-algorithm 挂载访问
# =============================================================================

FROM registry.intsig.net/pytorch_public/pytorch@sha256:0279f7aa29974bf64e61d0ff6e979b41a249b3662a46e30778dbf80b8c99c361

ENV DEBIAN_FRONTEND=noninteractive \
    # 禁止 Python 加载 ~/.local/lib 下的用户包，防止宿主机挂载目录干扰容器内的包版本
    PYTHONNOUSERSITE=1

# 切换 apt 源为阿里云（Ubuntu 22.04 Jammy）
RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
        /etc/apt/sources.list

# 系统依赖（PIL、OpenCV 等需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        git \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 配置 pip 使用清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
 && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# torchvision（base 镜像已含 torch，只需补装 torchvision，使用北京外国语大学 torch 镜像）
RUN pip install --no-cache-dir \
        "torchvision==0.18.0" \
        --index-url https://mirror.bfsu.edu.cn/pypi/web/simple \
        --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 项目依赖（不含 torch/torchvision）
# transformers 固定到 4.44.2：最后一个兼容 torch 2.3 的稳定版本
# 4.45+ 要求 torch >= 2.4，且存在 nn 未定义的 bug
RUN pip install --no-cache-dir \
        "transformers==4.44.2" \
        accelerate \
        safetensors \
        einops \
        sentencepiece \
        protobuf \
        modelscope \
        ftfy \
        pynvml \
        pandas \
        "imageio[ffmpeg]" \
        "controlnet-aux==0.0.7" \
        tqdm \
        Pillow \
        numpy

# 构建时验证 RAFT 可用（不依赖 GPU）
RUN python -c "from torchvision.models.optical_flow import raft_large; print('RAFT OK')"

# =============================================================================
# diffsynth 的完整版本位于挂载路径，不打包进镜像。
# 运行时通过 PYTHONPATH 让 Python 优先找到完整版。
# =============================================================================
ENV PYTHONPATH="/juicefs-algorithm/data/IPT/yuang_feng/DiffSynth-Studio:${PYTHONPATH}"

WORKDIR /workspace

CMD ["bash"]
