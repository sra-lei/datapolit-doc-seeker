# 基础镜像用官方 python slim：ghcr.io 的 uv 镜像在国内网络环境拉取失败（连接被阻断），
# 改为在构建期用 pip 引导安装 uv 工具本身（仅此一步走 pip；项目依赖仍由 uv sync --frozen 管理）
# 参考：https://docs.astral.sh/uv/guides/integration/docker/
FROM python:3.12-slim

WORKDIR /app

# 构建期系统依赖（pymilvus/grpcio 可能需要编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 用 pip 安装 uv 工具（uv 官方支持 pip 安装；PyPI 可达性比 ghcr.io 可靠）
RUN pip install --no-cache-dir "uv>=0.11"

# Docker 推荐构建参数：预编译字节码 + 拷贝链接（避免跨文件系统硬链接失败）
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# 先装依赖，利用镜像层缓存（仅 pyproject.toml / uv.lock 变更时失效）
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 再拷源码并安装项目本身（默认 editable 安装；dev 环境挂载 ./src:/app/src 时改动即时生效）
COPY . .
RUN uv sync --frozen --no-dev

# 非 root 运行：创建 app 用户并接管 /app（venv 与源码一并归属）
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app && \
    chown -R app:app /app
USER app

EXPOSE 8001

# 健康检查（slim 镜像无 curl，用 urllib 调 /v1/health）
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/v1/health', timeout=5).status==200 else 1)"]

CMD ["uvicorn", "docs_seeker.app:app", "--host", "0.0.0.0", "--port", "8001"]
