# ============================================================
# Qwen3-Txt
# 基础镜像：PyTorch 2.11 + CUDA 12.8 + cuDNN 9（Python 3.12）
# ============================================================

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# ── 系统依赖 ─────────────────────────────────────────────────
# libgomp1        → PyTorch OpenMP 多线程（基础镜像可能缺失）
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# ── 1. 服务依赖 ───────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "websockets>=12.0" \
    "transformers>=4.51.0" \
    --break-system-packages

# ── 4. 复制启动脚本 ───────────────────────────────────────────
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ── 5. 代码 ───
COPY codes/ /app/codes/

# ── 6. 复制模型文件 ────
COPY Qwen3-0.6B /models/Qwen3-0.6B

# 暴露端口
EXPOSE 8001/tcp

CMD ["/app/start.sh"]