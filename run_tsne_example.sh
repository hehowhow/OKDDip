#!/bin/bash

# t-SNE 可视化示例脚本
# 使用方法: bash run_tsne_example.sh

# 设置模型路径（请根据实际情况修改）
MODEL_PATH="./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth"
MODEL="resnet32"
DATASET="CIFAR10"
NUM_BRANCHES=4

echo "========================================="
echo "t-SNE Visualization Examples"
echo "========================================="
echo ""

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请修改脚本中的 MODEL_PATH 变量为你的模型路径"
    exit 1
fi

# 示例 1: 使用所有分支的平均特征（推荐）
echo "示例 1: 生成所有分支的平均特征 t-SNE 图..."
python plot_tsne.py \
    --model_path "$MODEL_PATH" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num_branches "$NUM_BRANCHES" \
    --max_samples 5000 \
    --output tsne_ensemble.png

echo ""
echo "✓ 完成！图像保存为: tsne_ensemble.png"
echo ""

# 示例 2: 使用 Leader 分支的特征
echo "示例 2: 生成 Leader 分支的特征 t-SNE 图..."
python plot_tsne.py \
    --model_path "$MODEL_PATH" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num_branches "$NUM_BRANCHES" \
    --branch_idx 3 \
    --max_samples 5000 \
    --output tsne_leader.png

echo ""
echo "✓ 完成！图像保存为: tsne_leader.png"
echo ""

# 示例 3: 比较所有分支（可选，会生成多个图）
read -p "是否要生成所有分支的对比图？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "示例 3: 生成所有分支的对比图..."
    for branch in 0 1 2 3; do
        echo "  处理分支 $branch..."
        python plot_tsne.py \
            --model_path "$MODEL_PATH" \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --num_branches "$NUM_BRANCHES" \
            --branch_idx "$branch" \
            --max_samples 5000 \
            --output "tsne_branch${branch}.png"
    done
    echo ""
    echo "✓ 完成！图像保存为: tsne_branch0.png, tsne_branch1.png, tsne_branch2.png, tsne_branch3.png"
fi

echo ""
echo "========================================="
echo "所有 t-SNE 可视化已完成！"
echo "========================================="
echo ""
echo "生成的文件："
ls -lh tsne*.png tsne*.pdf 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "提示："
echo "  - PNG 格式适合预览和演示"
echo "  - PDF 格式适合论文发表（矢量图）"
echo "  - 查看 TSNE_GUIDE.md 了解更多用法"
echo ""



