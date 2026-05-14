#!/bin/bash

# 启动主程序
python /app/codes/main.py

# 启动 Qwen3-TTS 的 Web Demo（必须前台运行，否则容器会退出）
# exec qwen-tts-demo /models/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000