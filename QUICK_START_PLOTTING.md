# 快速开始：绘制消融实验图表

## 🚀 三步快速生成图表

### 步骤1: 安装依赖

```bash
# 方法1: 使用安装脚本（推荐）
chmod +x setup_plotting.sh
./setup_plotting.sh

# 方法2: 手动安装
pip install matplotlib numpy --user
```

### 步骤2: 提取实验数据（可选）

如果你已经运行了实验并有JSON结果文件：

```bash
python extract_results_for_plotting.py
```

这会自动扫描 `CIFAR100/300/GL/` 目录下的所有 `test_best_metrics*.json` 文件，并提取准确率数据。

### 步骤3: 生成图表

```bash
# 生成基础对比图（推荐用于论文）
python plot_ablation_study.py

# 生成详细度量对比图（包含柱状图和趋势图）
python plot_dissimilarity_metrics_comparison.py

# 生成堆叠柱状图
python plot_stacked_bar.py
```

---

## 📊 三种图表的区别

### 1. `plot_ablation_study.py` - 基础对比
```
适用场景：论文中简单清晰地展示有/无相异度模块的效果
图表类型：分组柱状图
数据需求：2组数据（有/无模块）× 多个分支数
```

![示例布局]
```
准确率(%)
  80 ┤                    ██ 有模块
  75 ┤        ██ 无模块   ██
  70 ┤   ██   ██    ██    ██
     └────────────────────────
        2    3    4    5  分支数
```

### 2. `plot_dissimilarity_metrics_comparison.py` - 详细对比
```
适用场景：详细展示不同度量方式的性能差异
图表类型：分组柱状图 + 折线趋势图
数据需求：多种度量方式 × 多个分支数
```

![示例布局]
```
准确率(%)
  80 ┤  ██ ██ ██ ██ ██ ██   (每个x位置有6根柱子)
  75 ┤  ██ ██ ██ ██ ██ ██
  70 ┤  ██ ██ ██ ██ ██ ██
     └──────────────────────────
       Base W1 W2 Euc KL Cos  度量方式
```

### 3. `plot_stacked_bar.py` - 堆叠展示
```
适用场景：强调改进幅度，视觉突出提升效果
图表类型：堆叠柱状图
数据需求：基准值 + 提升值
```

![示例布局]
```
准确率(%)
  80 ┤   ┌──┐ ┌──┐ ┌──┐ ┌──┐
  75 ┤   │+1│ │+1│ │+2│ │+1│  提升
  70 ┤   │73│ │75│ │76│ │76│  基准
     └────────────────────────
        2    3    4    5  分支数
```

---

## 🎯 推荐工作流

### 情况1: 还没有跑实验
```bash
# 1. 跑实验（不同度量方式）
python train_GL.py --model resnet32 --dataset CIFAR100 --num_branches 4 --dissimilarity_metric wasserstein1
python train_GL.py --model resnet32 --dataset CIFAR100 --num_branches 4 --dissimilarity_metric wasserstein2
python train_GL.py --model resnet32 --dataset CIFAR100 --num_branches 4 --dissimilarity_metric euclidean
python train_GL.py --model resnet32 --dataset CIFAR100 --num_branches 4 --dissimilarity_metric kl
python train_GL.py --model resnet32 --dataset CIFAR100 --num_branches 4 --dissimilarity_metric cosine

# 2. 提取结果
python extract_results_for_plotting.py

# 3. 手动将提取的数据复制到绘图脚本的数据部分

# 4. 生成图表
python plot_ablation_study.py
```

### 情况2: 已经有实验结果
```bash
# 1. 提取现有结果
python extract_results_for_plotting.py

# 2. 查看输出，将数据代码复制到绘图脚本中

# 3. 生成图表
python plot_dissimilarity_metrics_comparison.py
```

### 情况3: 使用示例数据快速预览
```bash
# 直接运行（使用脚本内置的示例数据）
python plot_ablation_study.py
python plot_dissimilarity_metrics_comparison.py
python plot_stacked_bar.py
```

---

## 📝 如何修改数据

### 简单修改（直接编辑数组）

打开任意绘图脚本，找到 `# ==================== 数据部分 ====================`，直接修改：

```python
# 分支数
branch_numbers = [2, 3, 4, 5, 6]  # 添加6分支

# 没有相异度模块的准确率 (%)
accuracy_without_dissim = [72.5, 74.8, 76.2, 75.9, 75.5]  # 添加对应数据

# 加入相异度模块的准确率 (%)
accuracy_with_dissim = [73.2, 75.9, 77.8, 77.3, 77.0]  # 添加对应数据
```

### 高级修改（从CSV导入）

如果你有很多数据，可以从CSV文件导入：

```python
import pandas as pd

# 读取CSV
df = pd.read_csv('results.csv')

# 提取数据
branch_numbers = df['branches'].tolist()
accuracy_without_dissim = df['baseline_acc'].tolist()
accuracy_with_dissim = df['with_dissim_acc'].tolist()
```

---

## 🎨 常用自定义

### 修改图表标题
```python
ax.set_title('Your Custom Title Here', fontsize=15, fontweight='bold')
```

### 修改颜色主题
```python
# 蓝红配色（对比强烈）
colors = ['#3498db', '#e74c3c']

# 绿橙配色（色盲友好）
colors = ['#2ecc71', '#f39c12']

# 灰蓝配色（专业风格）
colors = ['#7f8c8d', '#34495e']
```

### 修改Y轴范围
```python
ax.set_ylim([70, 80])  # 设置为你的数据范围
```

### 输出特定尺寸（适配论文模板）
```python
# IEEE双栏格式
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# IEEE单栏格式
fig, ax = plt.subplots(figsize=(7, 4))

# A4横向
fig, ax = plt.subplots(figsize=(11, 8))
```

---

## 🐛 常见问题

**Q: ModuleNotFoundError: No module named 'matplotlib'**
```bash
A: 运行 pip install matplotlib numpy --user
```

**Q: 中文显示为方块**
```bash
A: 安装中文字体或在脚本中使用英文标签
   Linux: sudo apt-get install fonts-wqy-zenhei
```

**Q: 图表生成但不显示**
```bash
A: 检查是否在远程服务器上运行。可以只保存文件不显示：
   注释掉 plt.show() 这一行
```

**Q: 数据标签重叠**
```bash
A: 增加图表宽度或减少数据点：
   fig, ax = plt.subplots(figsize=(14, 6))  # 加宽
```

**Q: PDF文件太大**
```bash
A: 降低DPI或使用PNG：
   plt.savefig('output.png', dpi=150)
```

---

## 📤 输出文件说明

所有脚本都会生成两种格式：

- **PNG格式** (`.png`): 适用于PPT、网页、预览
  - 优点：兼容性好，文件小
  - 缺点：放大会模糊

- **PDF格式** (`.pdf`): 适用于论文、出版
  - 优点：矢量图，无限放大不失真
  - 缺点：文件稍大

**推荐使用场景：**
- 论文投稿：使用PDF
- 论文预览/检查：使用PNG（加载快）
- PPT演示：使用PNG（兼容性好）
- 海报打印：使用PDF（高质量）

---

## 💡 专业技巧

1. **批量生成多个版本**
```bash
for metric in wasserstein1 wasserstein2 euclidean kl cosine; do
    python plot_ablation_study.py --metric $metric
done
```

2. **自动添加时间戳**
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'ablation_{timestamp}.png')
```

3. **使用配置文件**
```python
import yaml
config = yaml.safe_load(open('plot_config.yaml'))
branch_numbers = config['branches']
accuracy_data = config['accuracies']
```

---

生成时间：2025-11-09
脚本版本：v1.0


