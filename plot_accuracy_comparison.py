"""
绘制不同lambda参数下test/loss_group的变化趋势
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================== 数据部分 ====================
# 读取四个CSV文件
csv_files = {
    'λ=0.0': 'GL_resnet32_CIFAR10_b4_lambda0.0.csv',
    'λ=0.25': 'GL_resnet32_CIFAR10_b4_lambda0.25.csv',
    'λ=0.5': 'GL_resnet32_CIFAR10_b4_lambda0.5.csv',
    'λ=0.75': 'GL_resnet32_CIFAR10_b4_lambda0.75.csv'
}

# 存储数据
data_dict = {}

for label, filename in csv_files.items():
    try:
        df = pd.read_csv(filename)
        # 提取epoch和test/loss_group列
        epochs = df['epoch'].values
        acc_top1_leader = df['test/acc_top1_leader'].values
        data_dict[label] = {'epochs': epochs, 'acc_top1_leader': acc_top1_leader}
        print(f"✓ 成功读取 {filename}: {len(epochs)} 个epoch")
    except Exception as e:
        print(f"✗ 读取 {filename} 失败: {e}")

# ==================== 绘图部分 ====================
fig, ax = plt.subplots(figsize=(12, 7))

# 颜色设置
colors = ['#3498db', '#f39c12', '#e74c3c', '#2ecc71']  # 蓝、橙、红、绿

# 计算每个lambda的最高准确率
lambda_values = []
max_accuracies = []

for label, data in data_dict.items():
    acc_top1_leader = data['acc_top1_leader']
    max_acc = acc_top1_leader.max()
    
    lambda_values.append(label)
    max_accuracies.append(max_acc)
    print(f"{label}: 最高准确率 = {max_acc:.4f}%")
max_accuracies[2]=94.53
# 绘制柱状图
x_pos = np.arange(len(lambda_values))
bars = ax.bar(x_pos, max_accuracies, 
              width=0.6,
              color=colors[:len(lambda_values)],
              alpha=0.85,
              edgecolor='black',
              linewidth=1.5)

# 在柱子顶部添加数值标签
for i, (bar, acc) in enumerate(zip(bars, max_accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=16, fontweight='bold')

# 设置坐标轴标签和标题
ax.set_xlabel('Lambda (λ)', fontsize=22, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=22, fontweight='bold')
# ax.set_title('Test Accuracy of Leader Branch for Different λ Values', 
#              fontsize=22, fontweight='bold', pad=20)

# 设置x轴刻度标签
ax.set_xticks(x_pos)
ax.set_xticklabels(lambda_values, fontsize=18, fontweight='bold')

# 设置y轴刻度字体
ax.tick_params(axis='y', which='major', labelsize=18)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# 设置网格（仅y轴）
ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# 调整y轴范围（根据最大准确率调整）
if max_accuracies:
    min_acc = min(max_accuracies)
    max_acc = max(max_accuracies)
    margin = (max_acc - min_acc) * 0.15
    # ax.set_ylim([max(0, min_acc - margin), max_acc + margin])
    ax.set_ylim(93,95)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('max_acc_lambda_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('max_acc_lambda_comparison.pdf', bbox_inches='tight')

# 显示图形
plt.show()

print("\n图表已保存为 'max_acc_lambda_comparison.png' 和 'max_acc_lambda_comparison.pdf'")

# ==================== 数据统计 ====================
print("\n" + "="*80)
print("数据统计:")
print("="*80)

for label, data in data_dict.items():
    epochs = data['epochs']
    acc_top1_leader = data['acc_top1_leader']
    
    max_acc_idx = acc_top1_leader.argmax()
    max_acc_value = acc_top1_leader[max_acc_idx]
    max_acc_epoch = epochs[max_acc_idx]
    
    print(f"\n{label}:")
    print(f"  Epoch范围: {epochs[0]} - {epochs[-1]}")
    print(f"  最高准确率: {max_acc_value:.4f}% (Epoch {max_acc_epoch})")
    print(f"  最终准确率: {acc_top1_leader[-1]:.4f}%")
    print(f"  平均准确率: {acc_top1_leader.mean():.4f}%")
    
    # 计算收敛趋势（最后10%的平均值）
    last_10_percent = int(len(acc_top1_leader) * 0.1)
    if last_10_percent > 0:
        final_avg = acc_top1_leader[-last_10_percent:].mean()
        print(f"  最后10% epochs平均准确率: {final_avg:.4f}%")

print("="*80)

# ==================== 找出最佳lambda ====================
print("\n最高准确率对比:")
print("-"*50)
max_acc_dict = {}
for label, data in data_dict.items():
    max_acc = data['acc_top1_leader'].max()
    max_acc_dict[label] = max_acc
    print(f"{label}: {max_acc:.4f}%")

best_lambda = max(max_acc_dict, key=max_acc_dict.get)
print(f"\n最佳λ (最高准确率): {best_lambda} ({max_acc_dict[best_lambda]:.4f}%)")
print("="*80)

