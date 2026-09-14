 # ====================== 马氏AI会话智能体 ======================
# 基于 python:3.11-slim 构建（与 CI 使用的版本一致），镜像轻量、启动快
# ⚠️ 不要改回 python:3.10-slim：该标签现指向 CPython 3.10.21，其标准库会让 pip
# 内部直接崩溃（sre_parse 的 TypeError / SIGSEGV），连 pip 自升级都无法运行
FROM python:3.11-slim

# 设置时区（时间工具 get_current_time 依赖时区数据库）
ENV TZ=Asia/Shanghai
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖清单并安装（利用 Docker 层缓存，代码变更时无需重装依赖）
COPY requirements.txt .
# 升级 pip / setuptools / wheel：镜像自带的 pip 24.0 解析依赖时会崩溃，需升到
# pip>=24.2（实测升级到 26.2.1 后，全部依赖可正常安装）
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
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
