 # ====================== 马氏AI会话智能体 ======================
# 基于 python:3.10-slim 构建，镜像轻量、启动快
FROM python:3.10-slim

# 设置时区（时间工具 get_current_time 依赖时区数据库）
ENV TZ=Asia/Shanghai
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖清单并安装（利用 Docker 层缓存，代码变更时无需重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 启动脚本赋予执行权限
RUN chmod +x entrypoint.sh

EXPOSE 8501

# 健康检查：Streamlit 内置健康端点
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).read()==b'ok' else 1)"

ENTRYPOINT ["./entrypoint.sh"]
