# 分层构建：依赖层缓存命中时，改代码不会触发重装
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 只拷贝运行期需要的代码，测试与缓存不进镜像
COPY app/ ./app/

# 以非 root 运行。容器默认 root，一旦被突破影响面更大
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查打 /health：编排系统据此判断实例是否真的可用，
# 而不只是「进程还活着」
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# workers=1：会话存在进程内存里，多 worker 会导致同一 session
# 被路由到不同进程而丢历史。要横向扩展需先把 SessionStore 换成 Redis。
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
