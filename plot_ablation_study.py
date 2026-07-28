"""
绘制消融实验结果的分组柱状图
对比不同分支数下，有/无相异度模块的准确率表现
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持（如果需要中文标签）con
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==================== 数据部分 ====================
# 分支数
branch_numbers = [3, 4, 5,6,7,8]

# 没有相异度模块的准确率 (%)
accuracy_without_dissim = [78.71835443037975, 78.7007911392405, 79.26226265822785, 78.89636075949367,  78.4315664556962,78.41178797468355]

# 加入相异度模块的准确率 (%)
# 这里可以是不同度量方式的最佳结果
accuracy_with_dissim = [78.47112341772151, 79.24580696202532, 
    79.48536392405063, 79.06669303797468, 78.77223101265823,78.68868670886076]

# ==================== 绘图部分 ====================
# 设置图形大小和风格
fig, ax = plt.subplots(figsize=(10, 6))

# # 设置柱子的宽度和位置
bar_width = 0.35
x_pos = np.arange(len(branch_numbers))

# 绘制分组柱状图
bars1 = ax.bar(x_pos - bar_width/2, accuracy_without_dissim, bar_width,
               label='OKDDIP(Origin)', 
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)

bars2 = ax.bar(x_pos + bar_width/2, accuracy_with_dissim, bar_width,
               label='OKDDIP(PHR)',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

# 在柱子顶部添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# 设置坐标轴标签和标题
ax.set_xlabel('Number of Branches', fontsize=22, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=22, fontweight='bold')
# ax.set_title('Ablation Study: Impact of PHR Module on Model Performance', 
#              fontsize=14, fontweight='bold', pad=20)

# 设置x轴刻度
ax.set_xticks(x_pos)
ax.set_xticklabels(branch_numbers)

# 设置坐标轴刻度字号和粗体
ax.tick_params(axis='both', which='major', labelsize=18)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# 设置y轴范围，让差异更明显
ax.set_ylim([77, 80])

# 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# 添加图例到右上角
legend = ax.legend(loc='upper right', fontsize=16, framealpha=0.95, 
                   edgecolor='black', fancybox=True, shadow=True)
# 设置图例文字为粗体
for text in legend.get_texts():
    text.set_fontweight('bold')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('ablation_study_dissimilarity.png', dpi=300, bbox_inches='tight')
plt.savefig('ablation_study_dissimilarity.pdf', bbox_inches='tight')

# 显示图形
plt.show()

print("图表已保存为 'ablation_study_dissimilarity.png' 和 'ablation_study_dissimilarity.pdf'")


# ==================== 如果需要堆叠柱状图 ====================
# 取消下面的注释来绘制堆叠柱状图

# fig2, ax2 = plt.subplots(figsize=(10, 6))

# 堆叠柱状图
# bars1 = ax2.bar(x_pos, accuracy_without_dissim, bar_width*2,
#                 label='OKDDip', 
#                 color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)

# bars2 = ax2.bar(x_pos, 
#                 [accuracy_with_dissim[i] - accuracy_without_dissim[i] for i in range(len(branch_numbers))], 
#                 bar_width*2,
#                 bottom=accuracy_without_dissim,
#                 label='OKDDip+PHR(Ours)',
#                 color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)

# ax2.set_xlabel('Number of Branches', fontsize=13, fontweight='bold')
# ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
# # ax2.set_title('Comparison of Different Num of branches', fontsize=14, fontweight='bold', pad=20)
# ax2.set_xticks(x_pos)
# ax2.set_xticklabels(branch_numbers)
# ax2.set_ylim([75, 80])  # 设置y轴范围
# ax2.legend(loc='upper right', fontsize=11)
# ax2.grid(axis='y', linestyle='--', alpha=0.3)
# ax2.set_axisbelow(True)
# plt.tight_layout()
# plt.savefig('ablation_study_stacked.png', dpi=300, bbox_inches='tight')
# plt.savefig('ablation_study_dissimilarity.pdf', bbox_inches='tight')
# plt.show()

