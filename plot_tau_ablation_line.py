"""
绘制温度参数τ的消融实验折线图
展示温度对准确率的影响趋势
"""

import matplotlib.pyplot as plt
import numpy as np

# ==================== 数据部分 ====================
# 不同的温度参数τ（数值型，用于绘制连续的折线）
tau_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

# 对应的准确率 (%)
accuracies = [73.73893987341773,74.18284810126582, 74.22229430379746,74.5030379746836, 74.37074367088607,74.35096518987342, 74.21251582278481]

# ==================== 绘图部分 ====================
fig, ax = plt.subplots(figsize=(10, 6))

# 使用索引位置来实现等间距显示
x_positions = np.arange(len(tau_values))

# 绘制折线图（使用索引位置）
line = ax.plot(x_positions, accuracies, marker='o', markersize=10, 
               linewidth=2.5, color='#3498db', label='Test Accuracy',
               markerfacecolor='#e74c3c', markeredgecolor='black', 
               markeredgewidth=1.5)

# 在每个点上添加数值标签
for x_pos, acc in zip(x_positions, accuracies):
    ax.text(x_pos, acc + 0.08, f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 标记最佳点
best_idx = np.argmax(accuracies)
best_tau = tau_values[best_idx]
best_acc = accuracies[best_idx]
ax.scatter(x_positions[best_idx], best_acc, s=300, c='gold', marker='*', 
           edgecolors='black', linewidths=2, zorder=5, 
           label=f'Best: τ={best_tau} ({best_acc:.2f}%)')

# 添加τ=1.0的参考线
tau_1_idx = tau_values.index(1.0)
tau_1_acc = accuracies[tau_1_idx]
ax.axhline(y=tau_1_acc, color='gray', linestyle='--', linewidth=1.5, 
           alpha=0.5, label=f'τ=1.0 (Standard): {tau_1_acc:.2f}%')
ax.axvline(x=x_positions[tau_1_idx], color='gray', linestyle='--', linewidth=1.5, alpha=0.3)

# 设置坐标轴标签和标题
ax.set_xlabel('Temperature Parameter τ', fontsize=22, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=22, fontweight='bold')
# ax.set_title('Temperature Ablation Study: Effect on Model Performance', 
            #  fontsize=15, fontweight='bold', pad=20)

# 设置x轴范围和刻度（使用索引位置，但标签显示实际tau值）
ax.set_xlim([-0.5, len(tau_values) - 0.5])
ax.set_xticks(x_positions)
ax.set_xticklabels([str(t) for t in tau_values])

# 设置坐标轴刻度字号和粗体
ax.tick_params(axis='both', which='major', labelsize=18)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# 设置y轴范围
ax.set_ylim([73.0, 75.0])

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# 添加区域标注（使用索引位置）
ax.text(0.5, 74.8, 'Sharper\nWeights', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(5.0, 74.8, 'Softer\nWeights', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

# 添加箭头标注（使用索引位置）
ax.annotate('', xy=(0, 74.65), xytext=(1.8, 74.65),
            arrowprops=dict(arrowstyle='<-', lw=2, color='blue', alpha=0.5))
ax.annotate('', xy=(6, 74.65), xytext=(2.8, 74.65),
            arrowprops=dict(arrowstyle='<-', lw=2, color='orange', alpha=0.5))

# 添加图例
legend = ax.legend(loc='lower right', fontsize=14, framealpha=0.95)
# 设置图例文字为粗体
for text in legend.get_texts():
    text.set_fontweight('bold')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('tau_ablation_line.png', dpi=300, bbox_inches='tight')
plt.savefig('tau_ablation_line.pdf', bbox_inches='tight')

# 显示图形
plt.show()

print("图表已保存为 'tau_ablation_line.png' 和 'tau_ablation_line.pdf'")

# 打印数据摘要
print("\n" + "="*70)
print("温度参数τ消融实验结果（详细分析）:")
print("="*70)
print(f"{'τ':^8} | {'Accuracy':^12} | {'Δ from τ=1.0':^15} | {'Description':^25}")
print("-"*70)

tau_1_acc = accuracies[tau_values.index(1.0)]
for tau, acc in zip(tau_values, accuracies):
    improvement = acc - tau_1_acc
    if tau < 1.0:
        desc = "More aggressive"
    elif tau == 1.0:
        desc = "Standard softmax"
    else:
        desc = "More uniform"
    
    marker = " ★" if acc == max(accuracies) else ""
    print(f"{tau:^8.2f} | {acc:^12.2f}% | {improvement:^15.2f}% | {desc:^25}{marker}")

print("="*70)
print(f"\n关键发现:")
print(f"  • 最佳温度: τ = {tau_values[best_idx]}")
print(f"  • 最高准确率: {max(accuracies):.2f}%")
print(f"  • 相比τ=1.0提升: {max(accuracies) - tau_1_acc:+.2f}%")
print(f"  • 温度范围: [{min(tau_values)}, {max(tau_values)}]")
print("="*70)

