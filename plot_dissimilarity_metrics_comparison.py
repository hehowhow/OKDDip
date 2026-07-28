"""
绘制不同相异度度量方式的对比图
展示 Wasserstein-1, Wasserstein-2, Euclidean, KL, Cosine 等不同度量的效果
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据部分 ====================
# 不同的度量方式
metrics = ['Baseline\n(No Dissim)', 'Wasserstein-1', 'Wasserstein-2', 
           'Euclidean', 'KL Divergence', 'Cosine']

# 不同分支数下的准确率 (%) - 示例数据
# 每一行代表一个度量方式，每一列代表一个分支数配置
accuracy_data = {
    'Branch-2': [72.5, 73.2, 72.8, 73.0, 72.9, 73.1],
    'Branch-3': [74.8, 75.9, 75.5, 75.7, 75.3, 75.6],
    'Branch-4': [76.2, 77.8, 77.5, 77.6, 77.3, 77.4],
    'Branch-5': [75.9, 77.3, 77.0, 77.1, 76.8, 77.0],
}

# ==================== 绘制分组柱状图 ====================
fig, ax = plt.subplots(figsize=(14, 7))

# 颜色方案
colors = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#2ecc71']

# 设置柱子的宽度和位置
bar_width = 0.15
x_pos = np.arange(len(metrics))

# 为每个分支数配置绘制一组柱子
branch_configs = list(accuracy_data.keys())
for i, (branch_config, accuracies) in enumerate(accuracy_data.items()):
    offset = (i - len(branch_configs)/2 + 0.5) * bar_width
    bars = ax.bar(x_pos + offset, accuracies, bar_width,
                   label=branch_config, 
                   color=colors[i % len(colors)], 
                   alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # 在柱子顶部添加数值（只对最高的几个显示）
    for bar in bars:
        height = bar.get_height()
        if height >= max(accuracies) - 0.3:  # 只标注接近最高值的
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

# 设置坐标轴
ax.set_xlabel('Dissimilarity Metric', fontsize=14, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Ablation Study: Comparison of Different Dissimilarity Metrics', 
             fontsize=15, fontweight='bold', pad=20)

ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim([70, 80])

# 添加网格
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# 添加图例到右上角
ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
          ncol=2, edgecolor='black', fancybox=True, shadow=True,
          title='Number of Branches', title_fontsize=11)

plt.tight_layout()
plt.savefig('dissimilarity_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('dissimilarity_metrics_comparison.pdf', bbox_inches='tight')
plt.show()

print("图表已保存为 'dissimilarity_metrics_comparison.png' 和 'dissimilarity_metrics_comparison.pdf'")


# ==================== 绘制折线图版本 ====================
fig2, ax2 = plt.subplots(figsize=(12, 7))

# 为每个度量方式绘制一条折线
branch_nums = [2, 3, 4, 5]
colors_line = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#2ecc71']
markers = ['o', 's', '^', 'D', 'v', 'p']

for i, metric in enumerate(metrics):
    metric_accuracies = [accuracy_data[f'Branch-{b}'][i] for b in branch_nums]
    ax2.plot(branch_nums, metric_accuracies, 
             marker=markers[i], markersize=10, linewidth=2.5,
             label=metric.replace('\n', ' '), 
             color=colors_line[i], alpha=0.8)

ax2.set_xlabel('Number of Branches', fontsize=14, fontweight='bold')
ax2.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
ax2.set_title('Performance Trend: Different Dissimilarity Metrics vs Number of Branches', 
              fontsize=15, fontweight='bold', pad=20)

ax2.set_xticks(branch_nums)
ax2.set_ylim([71, 79])
ax2.grid(True, linestyle='--', alpha=0.4)

# 图例
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95,
           ncol=2, edgecolor='black', fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('dissimilarity_metrics_trend.png', dpi=300, bbox_inches='tight')
plt.savefig('dissimilarity_metrics_trend.pdf', bbox_inches='tight')
plt.show()

print("趋势图已保存为 'dissimilarity_metrics_trend.png' 和 'dissimilarity_metrics_trend.pdf'")


# ==================== 打印数据表格 ====================
print("\n" + "="*80)
print("实验结果数据表格:")
print("="*80)
print(f"{'Metric':<20} | {'Branch-2':<10} | {'Branch-3':<10} | {'Branch-4':<10} | {'Branch-5':<10}")
print("-"*80)
for i, metric in enumerate(metrics):
    metric_name = metric.replace('\n', ' ')
    values = [accuracy_data[f'Branch-{b}'][i] for b in branch_nums]
    print(f"{metric_name:<20} | {values[0]:<10.2f} | {values[1]:<10.2f} | {values[2]:<10.2f} | {values[3]:<10.2f}")
print("="*80)

# 计算改进幅度
print("\n相对于Baseline的改进幅度 (%):")
print("="*80)
print(f"{'Metric':<20} | {'Branch-2':<10} | {'Branch-3':<10} | {'Branch-4':<10} | {'Branch-5':<10}")
print("-"*80)
for i in range(1, len(metrics)):  # 跳过baseline
    metric_name = metrics[i].replace('\n', ' ')
    improvements = []
    for b_idx, b in enumerate(branch_nums):
        baseline = accuracy_data[f'Branch-{b}'][0]
        current = accuracy_data[f'Branch-{b}'][i]
        improvement = current - baseline
        improvements.append(improvement)
    print(f"{metric_name:<20} | {improvements[0]:+10.2f} | {improvements[1]:+10.2f} | {improvements[2]:+10.2f} | {improvements[3]:+10.2f}")
print("="*80)


