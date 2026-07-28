"""
绘制不同相异度度量方式的消融实验柱状图
"""

import matplotlib.pyplot as plt
import numpy as np

# ==================== 数据部分 ====================
# 不同的相异度度量方式
metrics = ['Without PHR', 'Wasserstein-1', 'Wasserstein-2', 
           'Euclidean', 'KL Divergence', 'Cosine']

# 对应的准确率 (%)
accuracies = [73.9814082278481, 74.50196835443038, 74.31107594936708, 73.842958860759, 74.32653481012658, 74.25284810126582]

# ==================== 绘图部分 ====================
fig, ax = plt.subplots(figsize=(10, 6))

# 设置柱子位置和宽度
x_pos = np.arange(len(metrics))
bar_width = 0.6

# 为不同的度量方式设置不同颜色
colors = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#2ecc71']

# 绘制柱状图
bars = ax.bar(x_pos, accuracies, bar_width,
              color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

# 在柱子顶部添加数值标签
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=16, fontweight='bold')

# 设置坐标轴标签和标题
ax.set_xlabel('Dissimilarity Metric', fontsize=22, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=22, fontweight='bold')
# ax.set_title('Comparison of Different Dissimilarity Metrics', 
#              fontsize=15, fontweight='bold', pad=20)

# 设置x轴刻度
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, fontsize=11)
ax.tick_params(axis='x', which='major', labelsize=12)
ax.tick_params(axis='y', which='major', labelsize=18)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
# 设置y轴范围
ax.set_ylim([72, 75])

# 添加水平网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# 添加基线参考线（Baseline的准确率）
# baseline_acc = accuracies[0]
# ax.axhline(y=baseline_acc, color='#95a5a6', linestyle='--', linewidth=2, 
#            alpha=0.5, label=f'Baseline: {baseline_acc:.2f}%')

# 添加图例
# ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('dissimilarity_ablation.png', dpi=300, bbox_inches='tight')
plt.savefig('dissimilarity_ablation.pdf', bbox_inches='tight')

# 显示图形
plt.show()

print("图表已保存为 'dissimilarity_ablation.png' 和 'dissimilarity_ablation.pdf'")

# 打印数据摘要
# print("\n" + "="*60)
# print("实验结果摘要:")
# print("="*60)
# for metric, acc in zip(metrics, accuracies):
#     improvement = acc - baseline_acc
#     metric_name = metric.replace('\n', ' ')
#     print(f"{metric_name:25s} | Acc: {acc:6.2f}% | Improvement: {improvement:+6.2f}%")
# print("="*60)

