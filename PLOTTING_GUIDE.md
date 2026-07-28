# 消融实验绘图指南

本目录包含三个绘图脚本，用于展示相异度模块消融实验的结果。

## 📊 脚本说明

### 1. `plot_ablation_study.py` - 基础对比图
**用途：** 对比有/无相异度模块在不同分支数下的准确率

**特点：**
- 分组柱状图（Grouped Bar Chart）
- 横轴：分支数量
- 纵轴：准确率
- 两组数据：无模块 vs 有模块

**运行：**
```bash
python plot_ablation_study.py
```

**输出文件：**
- `ablation_study_dissimilarity.png`
- `ablation_study_dissimilarity.pdf`

---

### 2. `plot_dissimilarity_metrics_comparison.py` - 详细度量对比
**用途：** 对比不同相异度度量方式（Wasserstein-1/2、Euclidean、KL、Cosine）的效果

**特点：**
- 包含两种图表：
  - 分组柱状图：多度量方式 × 多分支数的对比
  - 折线图：展示不同度量方式随分支数变化的趋势
- 自动计算并打印改进幅度

**运行：**
```bash
python plot_dissimilarity_metrics_comparison.py
```

**输出文件：**
- `dissimilarity_metrics_comparison.png` (柱状图)
- `dissimilarity_metrics_comparison.pdf`
- `dissimilarity_metrics_trend.png` (折线图)
- `dissimilarity_metrics_trend.pdf`

---

### 3. `plot_stacked_bar.py` - 堆叠柱状图
**用途：** 展示基准准确率和相异度模块带来的提升

**特点：**
- 真正的堆叠柱状图（Stacked Bar Chart）
- 底部：基准准确率
- 顶部：相异度模块的提升
- 包含两个图表：
  - 不同分支数的堆叠对比
  - 不同度量方式的堆叠对比（固定分支数）

**运行：**
```bash
python plot_stacked_bar.py
```

**输出文件：**
- `stacked_bar_ablation.png`
- `stacked_bar_ablation.pdf`
- `stacked_bar_metrics.png`
- `stacked_bar_metrics.pdf`

---

## 📝 如何修改数据

### 修改准确率数据

在每个脚本中找到 `# ==================== 数据部分 ====================` 标记，修改对应的数据列表。

**示例（plot_ablation_study.py）：**
```python
# 分支数
branch_numbers = [2, 3, 4, 5]

# 没有相异度模块的准确率 (%)
accuracy_without_dissim = [72.5, 74.8, 76.2, 75.9]

# 加入相异度模块的准确率 (%)
accuracy_with_dissim = [73.2, 75.9, 77.8, 77.3]
```

**示例（plot_dissimilarity_metrics_comparison.py）：**
```python
# 不同的度量方式
metrics = ['Baseline\n(No Dissim)', 'Wasserstein-1', 'Wasserstein-2', 
           'Euclidean', 'KL Divergence', 'Cosine']

# 不同分支数下的准确率 (%)
accuracy_data = {
    'Branch-2': [72.5, 73.2, 72.8, 73.0, 72.9, 73.1],
    'Branch-3': [74.8, 75.9, 75.5, 75.7, 75.3, 75.6],
    'Branch-4': [76.2, 77.8, 77.5, 77.6, 77.3, 77.4],
    'Branch-5': [75.9, 77.3, 77.0, 77.1, 76.8, 77.0],
}
```

---

## 🎨 自定义图表样式

### 修改颜色
在脚本中找到 `colors` 变量，修改颜色代码：
```python
colors = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#2ecc71']
```

常用颜色代码：
- `#3498db` - 蓝色
- `#e74c3c` - 红色
- `#2ecc71` - 绿色
- `#f39c12` - 橙色
- `#9b59b6` - 紫色

### 修改图表大小
```python
fig, ax = plt.subplots(figsize=(10, 6))  # (宽度, 高度) 单位：英寸
```

### 修改DPI（分辨率）
```python
plt.savefig('output.png', dpi=300)  # 默认300，可改为150, 600等
```

---

## 📈 推荐使用场景

| 场景 | 推荐脚本 | 原因 |
|------|---------|------|
| 论文中展示有/无模块对比 | `plot_ablation_study.py` | 简洁清晰，便于对比 |
| 展示多种度量方式的详细对比 | `plot_dissimilarity_metrics_comparison.py` | 信息量大，包含趋势分析 |
| 强调模块带来的提升幅度 | `plot_stacked_bar.py` | 视觉上突出改进效果 |
| PPT展示 | 任意脚本 | 都输出高分辨率PDF格式 |

---

## 🛠️ 依赖安装

确保安装了必要的Python库：

```bash
pip install matplotlib numpy
```

如果需要中文支持：
```bash
# Linux
sudo apt-get install fonts-wqy-zenhei

# 或安装其他中文字体
```

---

## 💡 使用提示

1. **批量生成图表：** 可以将三个脚本合并到一个脚本中一次性生成所有图表
2. **实验数据管理：** 建议将实验结果保存为JSON文件，然后在绘图脚本中自动读取
3. **图表版本控制：** 每次实验后将生成的图表重命名，添加时间戳或实验ID
4. **论文格式：** 使用PDF格式输出，矢量图在论文中缩放不失真

---

## 📚 示例工作流

```bash
# 1. 运行实验（使用不同的相异度度量）
python train_GL.py --model resnet32 --dataset CIFAR100 --dissimilarity_metric wasserstein1
python train_GL.py --model resnet32 --dataset CIFAR100 --dissimilarity_metric wasserstein2
python train_GL.py --model resnet32 --dataset CIFAR100 --dissimilarity_metric euclidean
python train_GL.py --model resnet32 --dataset CIFAR100 --dissimilarity_metric kl
python train_GL.py --model resnet32 --dataset CIFAR100 --dissimilarity_metric cosine

# 2. 收集实验结果，更新绘图脚本中的数据

# 3. 生成图表
python plot_ablation_study.py
python plot_dissimilarity_metrics_comparison.py
python plot_stacked_bar.py

# 4. 检查生成的PNG和PDF文件
ls -lh *.png *.pdf
```

---

## 📞 问题排查

**问题：图表中文显示为方块**
- 解决：安装中文字体或修改脚本使用英文标签

**问题：图表太小/太大**
- 解决：调整 `figsize=(width, height)` 参数

**问题：数据标签重叠**
- 解决：减少数据点数量或增加图表宽度

**问题：生成的图片模糊**
- 解决：增加 `dpi` 参数值（如 `dpi=600`）

---

生成时间：2025-11-09


