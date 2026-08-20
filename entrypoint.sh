#!/bin/bash
# ====================== 应用启动脚本 ======================
# 职责：检查依赖完整性 → 初始化数据目录 → 启动 Streamlit
set -e

echo "==> 检查依赖是否安装完整..."
python -c "import streamlit, openai, chromadb, langchain_community, redis" 2>/dev/null || {
    echo "❌ 依赖缺失，请重新构建镜像：docker-compose build --no-cache"
    exit 1
}
echo "✅ 依赖检查通过"

echo "==> 初始化数据目录..."
# 挂载卷不存在时自动创建，防止写入报错
mkdir -p ./session_data ./chroma_db
echo "✅ 数据目录就绪（session_data / chroma_db）"

PORT="${PORT:-8501}"
echo "==> 启动 Streamlit 应用（端口 ${PORT}）..."
exec streamlit run 06.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false
