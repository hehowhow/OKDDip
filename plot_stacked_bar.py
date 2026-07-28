"""
绘制堆叠柱状图 (Stacked Bar Chart)
展示基准准确率和相异度模块带来的提升
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据部分 ====================
# 分支数
branch_numbers = [2, 3, 4, 5]

# 基准准确率（没有相异度模块）
baseline_accuracy = [72.5, 74.8, 76.2, 75.9]

# 相异度模块带来的提升
improvement_by_dissim = [0.7, 1.1, 1.6, 1.4]

# 加入相异度模块后的总准确率
total_accuracy = [baseline_accuracy[i] + improvement_by_dissim[i] 
                  for i in range(len(branch_numbers))]

# ==================== 绘制堆叠柱状图 ====================
fig, ax = plt.subplots(figsize=(10, 7))

bar_width = 0.5
x_pos = np.arange(len(branch_numbers))

# 底部：基准准确率
bars1 = ax.bar(x_pos, baseline_accuracy, bar_width,
               label='Baseline Accuracy (Without Dissimilarity)', 
               color='#3498db', alpha=0.85, edgecolor='black', linewidth=1)

# 顶部：相异度模块带来的提升
bars2 = ax.bar(x_pos, improvement_by_dissim, bar_width,
               bottom=baseline_accuracy,
               label='Improvement from Dissimilarity Module',
               color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1)

# 在柱子中间添加基准准确率数值
for i, (bar, val) in enumerate(zip(bars1, baseline_accuracy)):
    height = val / 2
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='center', fontsize=11, 
            fontweight='bold', color='white')

# 在堆叠部分添加提升数值
for i, (bar, val) in enumerate(zip(bars2, improvement_by_dissim)):
    height = baseline_accuracy[i] + val / 2
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'+{val:.1f}%',
            ha='center', va='center', fontsize=11, 
            fontweight='bold', color='white')

# 在柱子顶部添加总准确率
for i, (x, total) in enumerate(zip(x_pos, total_accuracy)):
    ax.text(x, total + 0.3,
            f'{total:.1f}%',
            ha='center', va='bottom', fontsize=12, 
            fontweight='bold', color='black')

# 设置坐标轴
ax.set_xlabel('Number of Branches', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Stacked Bar Chart: Impact of Dissimilarity Module\n(Baseline + Improvement)', 
             fontsize=15, fontweight='bold', pad=20)

ax.set_xticks(x_pos)
ax.set_xticklabels(branch_numbers, fontsize=12)
ax.set_ylim([0, 82])

# 添加水平网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# 添加图例到右上角
ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
          edgecolor='black', fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('stacked_bar_ablation.png', dpi=300, bbox_inches='tight')
plt.savefig('stacked_bar_ablation.pdf', bbox_inches='tight')
plt.show()

print("堆叠柱状图已保存为 'stacked_bar_ablation.png' 和 'stacked_bar_ablation.pdf'")


# ==================== 另一种堆叠方式：不同度量方式的对比 ====================
fig2, ax2 = plt.subplots(figsize=(12, 7))

# 数据：不同度量方式在4分支下的表现
metrics_short = ['Baseline', 'W-1', 'W-2', 'Euclid', 'KL', 'Cosine']
base_acc = 76.2  # 4分支的基准准确率

# 不同度量方式的提升
improvements = {
    'Baseline': 0,
    'W-1': 1.6,
    'W-2': 1.3,
    'Euclid': 1.4,
    'KL': 1.1,
    'Cosine': 1.2
}

base_values = [base_acc] * len(metrics_short)
improvement_values = [improvements[m] for m in metrics_short]
total_values = [base_acc + improvements[m] for m in metrics_short]

x_pos2 = np.arange(len(metrics_short))

# 绘制堆叠柱状图
bars1 = ax2.bar(x_pos2, base_values, 0.6,
                label='Baseline Accuracy', 
                color='#3498db', alpha=0.85, edgecolor='black', linewidth=1)

bars2 = ax2.bar(x_pos2, improvement_values, 0.6,
                bottom=base_values,
                label='Improvement',
                color=['#95a5a6', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c'],
                alpha=0.85, edgecolor='black', linewidth=1)

# 添加总值标签
for i, (x, total) in enumerate(zip(x_pos2, total_values)):
    ax2.text(x, total + 0.2,
             f'{total:.1f}%',
             ha='center', va='bottom', fontsize=11, 
             fontweight='bold')

ax2.set_xlabel('Dissimilarity Metric', fontsize=14, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax2.set_title('Stacked Bar Chart: Comparison of Different Metrics (4 Branches)', 
              fontsize=15, fontweight='bold', pad=20)

ax2.set_xticks(x_pos2)
ax2.set_xticklabels(metrics_short, fontsize=12)
ax2.set_ylim([0, 82])

ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
ax2.set_axisbelow(True)

ax2.legend(loc='upper right', fontsize=11, framealpha=0.95,
           edgecolor='black', fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('stacked_bar_metrics.png', dpi=300, bbox_inches='tight')
plt.savefig('stacked_bar_metrics.pdf', bbox_inches='tight')
plt.show()

print("度量对比堆叠图已保存为 'stacked_bar_metrics.png' 和 'stacked_bar_metrics.pdf'")

# 打印数据摘要
print("\n" + "="*60)
print("数据摘要:")
print("="*60)
print(f"{'Branches':<12} {'Baseline':<12} {'Improvement':<12} {'Total':<12}")
print("-"*60)
for i, b in enumerate(branch_numbers):
    print(f"{b:<12} {baseline_accuracy[i]:<12.2f} {improvement_by_dissim[i]:<12.2f} {total_accuracy[i]:<12.2f}")
print("="*60)


