"""
绘制温度参数τ的消融实验柱状图
展示不同温度对softmax归一化的影响
"""

import matplotlib.pyplot as plt
import numpy as np

# ==================== 数据部分 ====================
# 不同的温度参数τ
tau_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

# 对应的准确率 (%)
accuracies = [73.73893987341773,74.18284810126582, 74.22229430379746,74.5030379746836, 74.37074367088607,74.35096518987342, 74.21251582278481]

# ==================== 绘图部分 ====================
fig, ax = plt.subplots(figsize=(10, 6))

# 设置柱子位置和宽度
x_pos = np.arange(len(tau_values))
bar_width = 0.6

# 设置颜色渐变（从蓝到红，表示从低温到高温）
colors = ['#2980b9', '#3498db', '#5dade2', '#f39c12', '#e67e22', '#e74c3c']

# 绘制柱状图
bars = ax.bar(x_pos, accuracies, bar_width,
              color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

# 在柱子顶部添加数值标签
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# 设置坐标轴标签和标题
ax.set_xlabel('Temperature Parameter τ', fontsize=22, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=22, fontweight='bold')
# ax.set_title('Ablation Study: Impact of Temperature on Softmax Normalization', 
#              fontsize=15, fontweight='bold', pad=20)

# 设置x轴刻度
ax.set_xticks(x_pos)
ax.set_xticklabels(tau_values, fontsize=12)

# 设置y轴范围
ax.set_ylim([73, 75])

# 添加水平网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# 添加注释说明不同温度的效果
ax.text(0.98, 0.98, 'Lower τ → Sharper distribution\nHigher τ → Softer distribution', 
        transform=ax.transAxes, fontsize=18, verticalalignment='top',
        horizontalalignment='right', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# 标记最佳τ值
# best_idx = np.argmax(accuracies)
# ax.scatter(best_idx, accuracies[best_idx], s=200, c='gold', marker='*', 
#            edgecolors='black', linewidths=2, zorder=5, label=f'Best: τ={tau_values[best_idx]}')

ax.tick_params(axis='x', which='major', labelsize=18)
ax.tick_params(axis='y', which='major', labelsize=18)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
# 添加图例
# ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
# legend = ax.legend(loc='upper right', fontsize=16, framealpha=0.95, 
#                    edgecolor='black', fancybox=True, shadow=True)
# # 设置图例文字为粗体
# for text in legend.get_texts():
#     text.set_fontweight('bold')
# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('tau_ablation.png', dpi=300, bbox_inches='tight')
plt.savefig('tau_ablation.pdf', bbox_inches='tight')

# 显示图形
plt.show()

print("图表已保存为 'tau_ablation.png' 和 'tau_ablation.pdf'")

# 打印数据摘要
# print("\n" + "="*60)
# print("温度参数τ消融实验结果:")
# print("="*60)
# baseline_acc = accuracies[2]  # τ=1.0作为baseline
# for tau, acc in zip(tau_values, accuracies):
#     improvement = acc - baseline_acc
#     print(f"τ = {tau:5s} | Acc: {acc:6.2f}% | Δ from τ=1.0: {improvement:+6.2f}%")
# print("="*60)
# print(f"最佳温度: τ = {tau_values[best_idx]} (准确率: {accuracies[best_idx]:.2f}%)")
# print("="*60)

