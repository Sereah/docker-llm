# ============================================================
# Qwen3-TTS 云端服务
# 基础镜像：PyTorch 2.11 + CUDA 12.8 + cuDNN 9（Python 3.12）
# ============================================================

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# ── 系统依赖 ─────────────────────────────────────────────────
# libsndfile1     → soundfile / librosa 读写音频文件的底层 C 库
# libgomp1        → PyTorch OpenMP 多线程（基础镜像可能缺失）
# curl            → HEALTHCHECK 探针
# ffmpeg / sox    → 音频处理
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
        curl \
        ffmpeg \
        sox \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# ── 1. flash-attn（先装，下载耗时最长，放最前利用层缓存） ──
# 注意：此 wheel 是 Python 3.12 + CUDA 12 + Torch 2.11 + cxx11abi=TRUE
# 若你的环境不同，在 https://github.com/lesj0610/flash-attention/releases 找对应 wheel
RUN pip install --no-cache-dir \
    "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    --no-build-isolation --break-system-packages

# ── 2. 服务依赖 ───────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "websockets>=12.0" \
    "pydantic>=2.7.0" \
    "pydantic-settings>=2.3.0" \
    "loguru>=0.7.2" \
    "scipy>=1.13.0" \
    --break-system-packages

# ── 3. qwen-tts 及其官方依赖链 ───────────────────────────────
# qwen-tts 会自动安装：transformers==4.57.3, accelerate==1.12.0,
# librosa, torchaudio, soundfile, einops, onnxruntime 等
RUN pip install --no-cache-dir -U qwen-tts --break-system-packages

# ── 4. 复制启动脚本 ───────────────────────────────────────────
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ── 5. 代码（开发时不 COPY，改用 Volume 挂载；生产时打开） ───
# 开发模式：注释掉下面这段，在 docker-compose 里用 volumes 挂载
# 生产模式：取消注释，代码打进镜像
COPY codes/ /app/codes/

# ── 6. 复制模型文件（体积最大，放最后，改代码不重复复制） ────
COPY Qwen3-TTS-12Hz-1.7B-CustomVoice /models/Qwen3-TTS-12Hz-1.7B-CustomVoice

# ── 环境变量 ──────────────────────────────────────────────────
ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

# 暴露端口
EXPOSE 8000/tcp

CMD ["/app/start.sh"]