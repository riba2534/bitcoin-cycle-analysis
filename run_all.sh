#!/usr/bin/env bash
set -euo pipefail

# Bitcoin Cycle Analysis - 一键运行脚本
# 依次执行: 数据获取 → 主分析 → 专业图表 → 预测图表

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "  Bitcoin Cycle Analysis Pipeline"
echo "=========================================="
echo ""

# 检查 Python3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3，请先安装 Python 3.8+${NC}"
    exit 1
fi

# 检查依赖
echo "检查 Python 依赖..."
python3 -c "import requests, pandas, numpy, matplotlib, scipy" 2>/dev/null || {
    echo -e "${YELLOW}正在安装依赖...${NC}"
    pip3 install -r requirements.txt
}
echo -e "${GREEN}依赖检查通过${NC}"
echo ""

# Step 1: 数据获取（可选跳过）
if [ -d "data" ] && [ -f "data/btcusdt_1d.csv" ]; then
    echo -e "${YELLOW}[1/4] 已检测到本地数据，跳过数据获取${NC}"
    echo "      如需重新获取，请删除 data/ 目录后重新运行"
else
    echo -e "[1/4] 从币安 API 获取数据..."
    python3 fetch_binance_data.py
    echo -e "${GREEN}[1/4] 数据获取完成${NC}"
fi
echo ""

# Step 2: 主分析 + 图表 01-12
echo "[2/4] 运行主分析，生成图表 01-12..."
python3 analyze_btc_cycle.py
echo -e "${GREEN}[2/4] 主分析完成${NC}"
echo ""

# Step 3: 专业图表 13-15
echo "[3/4] 生成专业图表 13-15..."
python3 generate_pro_charts.py
echo -e "${GREEN}[3/4] 专业图表完成${NC}"
echo ""

# Step 4: 预测图表 16-18
echo "[4/4] 生成预测图表 16-18..."
python3 generate_forecast_charts.py
echo -e "${GREEN}[4/4] 预测图表完成${NC}"
echo ""

echo "=========================================="
echo -e "${GREEN}  全部完成! 共生成 18 张图表${NC}"
echo "  图表目录: charts/"
echo "  研究报告: REPORT.md"
echo "=========================================="
