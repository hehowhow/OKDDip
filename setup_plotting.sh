#!/bin/bash

# 安装绘图所需的依赖
echo "正在安装绘图所需的Python库..."

# 尝试使用pip3
if command -v pip3 &> /dev/null; then
    pip3 install matplotlib numpy --user
elif command -v pip &> /dev/null; then
    pip install matplotlib numpy --user
else
    echo "错误: 找不到pip或pip3命令"
    echo "请手动安装: pip install matplotlib numpy"
    exit 1
fi

echo ""
echo "安装完成！"
echo ""
echo "现在可以运行绘图脚本："
echo "  python plot_ablation_study.py"
echo "  python plot_dissimilarity_metrics_comparison.py"
echo "  python plot_stacked_bar.py"
echo ""


