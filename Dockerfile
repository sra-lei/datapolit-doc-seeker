FROM python:3.12-slim

WORKDIR /app

# 构建期系统依赖（pymilvus/grpcio 可能需要编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 先装依赖，利用镜像层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# 非 root 运行：创建 app 用户并接管 /app（editable 安装产物一并归属）
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app && \
    chown -R app:app /app
USER app

EXPOSE 8001

# 健康检查（python:3.12-slim 无 curl，用 urllib 调 /v1/health）
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/v1/health', timeout=5).status==200 else 1)"]

CMD ["uvicorn", "docs_seeker.app:app", "--host", "0.0.0.0", "--port", "8001"]
