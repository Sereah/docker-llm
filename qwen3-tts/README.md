### 模型

https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

### 下载模型到当前目录
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice.git

### 打包docker
``` shell
# 1. 导出为 tar 文件
docker save -o qwen3-tts-1.7b.tar qwen3-tts-1.7b

# 2. 和 docker-compose.yml 一起打包
tar czf qwen3-tts-dist.tar.gz qwen3-tts-1.7b.tar docker-compose.yml
```
