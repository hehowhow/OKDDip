"""
绘制不同lambda参数下test/loss_group的变化趋势
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================== 数据部分 ====================
# 读取四个CSV文件
csv_files = {
    'λ=0.0': 'GL_resnet32_CIFAR10_b4_lambda0.75.csv',
    'λ=0.25': 'GL_resnet32_CIFAR10_b4_lambda0.5.csv',
    'λ=0.5': 'GL_resnet32_CIFAR10_b4_lambda0.25.csv',
    'λ=0.75': 'GL_resnet32_CIFAR10_b4_lambda0.0.csv'
}

# 存储数据
data_dict = {}

for label, filename in csv_files.items():
    try:
        df = pd.read_csv(filename)
        # 提取epoch和test/loss_group列
        epochs = df['epoch'].values
        diversity = df['test/diversity'].values
        data_dict[label] = {'epochs': epochs, 'diversity': diversity}
        print(f"✓ 成功读取 {filename}: {len(epochs)} 个epoch")
    except Exception as e:
        print(f"✗ 读取 {filename} 失败: {e}")

# ==================== 绘图部分 ====================
fig, ax = plt.subplots(figsize=(12, 7))

# 颜色和线型设置
colors = ['#3498db', '#f39c12', '#e74c3c', '#2ecc71']  # 蓝、橙、红、绿
# markers = ['o', 's', '^']
line_styles = ['-', '-', '-', '-']

# 绘制四条折线
for idx, (label, data) in enumerate(data_dict.items()):
    epochs = data['epochs']
    diversity = data['diversity']
    
    # 每隔一定间隔显示marker，避免太密集
    # marker_interval = max(1, len(epochs) // 20)
    
    ax.plot(epochs, diversity, 
            label=label,
            color=colors[idx],
            linestyle=line_styles[idx],
            linewidth=3.2,
            # marker=markers[idx],
            # markevery=marker_interval,
            # markersize=6,
            # markerfacecolor=colors[idx],
            # markeredgecolor='black',
            # markeredgewidth=0.5,
            alpha=0.85)

# 设置坐标轴标签和标题
ax.set_xlabel('Epoch', fontsize=22, fontweight='bold')
ax.set_ylabel('Test Diversity', fontsize=22, fontweight='bold')
# ax.set_title('Test Diversity vs Epoch for Different λ Values', 
#              fontsize=22, fontweight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=18)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
# 设置网格
ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# 添加图例
ax.legend(loc='upper right', fontsize=16, framealpha=0.95,
          edgecolor='black', fancybox=True, shadow=True,
          prop={'weight': 'bold', 'size': 16})

# 调整y轴范围（根据数据自动调整）
if data_dict:
    all_losses = []
    for data in data_dict.values():
        all_losses.extend(data['diversity'])
    
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    margin = (max_loss - min_loss) * 0.1
    ax.set_ylim([max(0, min_loss - margin), max_loss + margin])

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('diversity_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('diversity_comparison.pdf', bbox_inches='tight')

# 显示图形
plt.show()

print("\n图表已保存为 'diversity_comparison.png' 和 'diversity_comparison.pdf'")

# ==================== 数据统计 ====================
print("\n" + "="*80)
print("数据统计:")
print("="*80)

for label, data in data_dict.items():
    epochs = data['epochs']
    diversity = data['diversity']
    
    print(f"\n{label}:")
    print(f"  Epoch范围: {epochs[0]} - {epochs[-1]}")
    print(f"  Loss Group范围: {diversity.min():.6f} - {diversity.max():.6f}")
    print(f"  最终Loss Group: {diversity[-1]:.6f}")
    print(f"  平均Loss Group: {diversity.mean():.6f}")
    
    # 计算收敛趋势（最后10%的平均值）
    last_10_percent = int(len(diversity) * 0.1)
    if last_10_percent > 0:
        final_avg = diversity[-last_10_percent:].mean()
        print(f"  最后10% epochs平均Loss: {final_avg:.6f}")

print("="*80)

# ==================== 找出最佳lambda ====================
print("\n最终Loss Group对比:")
print("-"*50)
final_losses = {}
for label, data in data_dict.items():
    final_loss = data['diversity'][-1]
    final_losses[label] = final_loss
    print(f"{label}: {final_loss:.6f}")

best_lambda = min(final_losses, key=final_losses.get)
print(f"\n最佳λ (最低final loss): {best_lambda}")
print("="*80)

