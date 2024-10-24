# Python 3.9를 기본 이미지로 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Start Generation Here
RUN git clone https://github.com/kiyoungchoi/tgrad.git && \
    cd tgrad && \
    pip install --no-cache-dir -r requirements.txt
# End Generation Here

# Python 패키지 설치
# COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
