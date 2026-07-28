# t-SNE 可视化指南

本指南说明如何使用 `plot_tsne.py` 脚本为训练好的模型生成 t-SNE 可视化图。

## 功能说明

`plot_tsne.py` 脚本可以：
1. 加载训练好的模型权重（`best.pth`）
2. 从测试集中提取特征（可以选择特定分支或所有分支的平均特征）
3. 使用 t-SNE 将高维特征降维到 2D
4. 绘制彩色散点图，不同类别用不同颜色表示

## 安装依赖

```bash
pip install scikit-learn matplotlib
```

## 基本使用

### 示例 1: 使用所有分支的平均特征（推荐）

```bash
python plot_tsne.py \
    --model_path ./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth \
    --model resnet32 \
    --dataset CIFAR10 \
    --num_branches 4 \
    --output tsne_ensemble.png
```

### 示例 2: 使用特定分支的特征

```bash
# 使用第0个分支的特征
python plot_tsne.py \
    --model_path ./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth \
    --model resnet32 \
    --dataset CIFAR10 \
    --num_branches 4 \
    --branch_idx 0 \
    --output tsne_branch0.png

# 使用第3个分支（leader branch）的特征
python plot_tsne.py \
    --model_path ./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth \
    --model resnet32 \
    --dataset CIFAR10 \
    --num_branches 4 \
    --branch_idx 3 \
    --output tsne_leader.png
```

### 示例 3: CIFAR-100 数据集

```bash
python plot_tsne.py \
    --model_path ./CIFAR100/300/GL/resnet32B4T3.0SKLV0/best.pth \
    --model resnet32 \
    --dataset CIFAR100 \
    --num_branches 4 \
    --max_samples 10000 \
    --output tsne_cifar100.png
```

## 参数说明

### 必需参数

- `--model_path`: 模型权重文件路径（例如 `./CIFAR10/300/GL/.../best.pth`）

### 可选参数

- `--model`: 模型架构（默认: `resnet32`）
  - 可选: `resnet32`, `resnet110`, `vgg16`, `vgg19`, 等
  
- `--dataset`: 数据集名称（默认: `CIFAR10`）
  - 可选: `CIFAR10`, `CIFAR100`
  
- `--num_branches`: 模型的分支数量（默认: `4`）

- `--branch_idx`: 要提取特征的分支索引（默认: `None`）
  - `None`: 使用所有分支的平均特征（推荐）
  - `0, 1, 2`: 使用特定peer分支的特征
  - `3`: 使用leader分支的特征（当 `num_branches=4` 时）

- `--batch_size`: 数据加载的批大小（默认: `128`）

- `--max_samples`: 使用的最大样本数（默认: `5000`）
  - 设置为 `-1` 表示使用全部测试集数据
  - 较小的值可以加速计算，较大的值可以获得更全面的可视化

- `--perplexity`: t-SNE 的困惑度参数（默认: `30`）
  - 典型范围: 5-50
  - 较小的值关注局部结构，较大的值关注全局结构

- `--n_iter`: t-SNE 的迭代次数（默认: `1000`）
  - 更多的迭代可能获得更好的结果，但需要更长时间

- `--gpu_id`: GPU ID（默认: `0`）

- `--output`: 输出文件路径（默认: `tsne_visualization.png`）
  - 会同时生成 PNG 和 PDF 两种格式

## 高级用法

### 比较不同分支的特征分布

```bash
# 生成所有分支的t-SNE图进行比较
for branch in 0 1 2 3; do
    python plot_tsne.py \
        --model_path ./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth \
        --model resnet32 \
        --dataset CIFAR10 \
        --num_branches 4 \
        --branch_idx $branch \
        --output tsne_branch${branch}.png
done

# 生成ensemble特征的t-SNE图
python plot_tsne.py \
    --model_path ./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth \
    --model resnet32 \
    --dataset CIFAR10 \
    --num_branches 4 \
    --output tsne_ensemble.png
```

### 使用全部测试集数据（更精确但更慢）

```bash
python plot_tsne.py \
    --model_path ./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth \
    --model resnet32 \
    --dataset CIFAR10 \
    --num_branches 4 \
    --max_samples -1 \
    --perplexity 50 \
    --n_iter 2000 \
    --output tsne_full.png
```

### 调整 t-SNE 参数以获得更好的可视化效果

```bash
# 关注局部结构
python plot_tsne.py \
    --model_path ./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth \
    --perplexity 10 \
    --output tsne_local.png

# 关注全局结构
python plot_tsne.py \
    --model_path ./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth \
    --perplexity 50 \
    --output tsne_global.png
```

## 输出说明

脚本会生成：
1. PNG 格式的高分辨率图像（300 DPI）
2. PDF 格式的矢量图（适合论文发表）

图表特点：
- CIFAR-10: 10种颜色代表10个类别，带图例
- CIFAR-100: 100种颜色代表100个类别，图例在右侧
- 坐标轴标签加粗，字体大小适中
- 半透明散点，便于观察重叠区域

## 解释 t-SNE 结果

- **紧密聚类**: 同一类别的样本在特征空间中相似
- **分离良好**: 不同类别之间有明显边界，表示模型学到了区分性特征
- **混合区域**: 某些类别重叠，可能表示这些类别在视觉上相似（如猫和狗）

## 性能优化建议

1. **首次测试**: 使用默认的 `max_samples=5000` 快速查看效果
2. **最终可视化**: 使用 `max_samples=-1` 获得最准确的结果
3. **调整困惑度**: 如果聚类效果不佳，尝试 `perplexity` 在 10-50 之间的不同值
4. **增加迭代次数**: 如果图像还在变化，增加 `n_iter` 到 2000 或更多

## 故障排除

### 问题 1: 内存不足
```bash
# 减少样本数
python plot_tsne.py --max_samples 2000 ...
```

### 问题 2: t-SNE 运行太慢
```bash
# 减少样本数和迭代次数
python plot_tsne.py --max_samples 3000 --n_iter 500 ...
```

### 问题 3: 模型加载失败
```bash
# 确保模型路径正确，并且 num_branches 参数与训练时一致
```

## 示例输出

运行脚本后，终端会显示：
```
Using device: cuda
Loading CIFAR10 dataset...
Data loaded successfully.
Creating resnet32 model with 4 branches...
Loading model weights from ./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth...
Model loaded successfully.

Extracting features from All Branches (Average)...
Using 5000 samples from test set
Extracted features shape: (5000, 64)
Labels shape: (5000,)
Running t-SNE with perplexity=30, n_iter=1000...
[t-SNE] Computing pairwise distances...
[t-SNE] Iteration 50: ...
...
t-SNE completed. Plotting...
t-SNE visualization saved to tsne_visualization.png and PDF version

Done!
```

## 引用和参考

如果在论文中使用 t-SNE 可视化，请引用：
```
van der Maaten, L., & Hinton, G. (2008). 
Visualizing data using t-SNE. 
Journal of machine learning research, 9(11).
```



