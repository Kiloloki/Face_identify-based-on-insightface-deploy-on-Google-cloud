# FROM python:3.10-slim

# WORKDIR /app

# COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 8080

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


# FROM python:3.10-slim

# WORKDIR /app

# COPY . /app

# # 加上构建工具！这两行千万别省略
# RUN apt-get update && apt-get install -y build-essential && apt-get clean

# RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 8080

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


FROM python:3.10-slim

WORKDIR /app

COPY . /app

# # 必须装常用依赖！
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     build-essential \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     libgl1 && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# EXPOSE 8080

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


# 使用 Python 官方镜像
FROM python:3.10-slim

# 系统依赖，face_recognition 用 dlib 需要这几个包
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app

# 拷贝项目文件
COPY . .

# 安装依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
