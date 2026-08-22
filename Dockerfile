FROM python:3.12-slim

WORKDIR /app

# 系统依赖（jieba 不需要编译，但 pymilvus 可能需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8001

CMD ["uvicorn", "docs_seeker.app:app", "--host", "0.0.0.0", "--port", "8001"]
