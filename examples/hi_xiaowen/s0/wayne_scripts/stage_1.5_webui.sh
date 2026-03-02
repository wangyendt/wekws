#!/bin/bash
# Stage 1.5 - 启动音频数据浏览器 WebUI
# 作者: Wayne
# 功能: 启动交互式 Web 界面，浏览、搜索和可视化音频数据

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$PROJECT_DIR")")")"

echo "=========================================="
echo "Stage 1.5 - 音频数据浏览器 WebUI"
echo "=========================================="
echo ""
echo "📂 仓库根目录: $REPO_ROOT"
echo "📂 项目目录: $PROJECT_DIR"
echo ""

# 检查数据库是否存在
DB_PATH="$PROJECT_DIR/data/metadata.db"
MODEL_REGISTRY_PATH="$PROJECT_DIR/wayne_scripts/webui_models.yaml"
if [ ! -f "$DB_PATH" ]; then
    echo "❌ 数据库不存在！"
    echo ""
    echo "请先运行以下命令构建数据库："
    echo "  cd $PROJECT_DIR"
    echo "  bash run_fsmn_ctc.sh 1.5 1.5"
    echo ""
    exit 1
fi

# 显示数据库信息
DB_SIZE=$(du -h "$DB_PATH" | cut -f1)
echo "✅ 数据库: $DB_PATH ($DB_SIZE)"
echo ""

# 检查 streamlit 是否安装
if ! command -v streamlit &> /dev/null; then
    echo "❌ streamlit 未安装！"
    echo ""
    echo "请运行以下命令安装："
    echo "  pip install streamlit pandas pillow"
    echo ""
    exit 1
fi

echo "✅ 依赖检查通过"
echo ""

# 启动 WebUI
echo "🚀 启动 WebUI..."
echo ""
echo "浏览器将自动打开，或手动访问: http://localhost:8501"
echo "模型下拉列表持久化文件: $MODEL_REGISTRY_PATH"
echo "按 Ctrl+C 停止服务"
echo "=========================================="
echo ""

cd "$REPO_ROOT"
streamlit run tools/webui_audio_explorer.py
