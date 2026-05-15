### 模型

https://www.modelscope.cn/models/Qwen/Qwen3-0.6B

### 下载模型到当前目录
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen3-0.6B.git

### 打包docker
``` shell
# 1. 导出为 tar 文件
docker save -o qwen3-txt-0.6b.tar qwen3-txt-0.6b

# 2. 和 docker-compose.yml 一起打包
tar czf qwen3-txt-dist.tar.gz qwen3-txt-0.6b.tar docker-compose.yml
```
