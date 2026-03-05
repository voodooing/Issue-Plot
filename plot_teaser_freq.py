import matplotlib.pyplot as plt
import pandas as pd

# 1. 准备数据
data = {
    "Method": ["SMFP(Ours)", "QVPO", "DIME", "DIPO", "FPMD-R", "FPMD-M", "TD3", "SAC", "PPO", "SPO", "Crossq", "MaxEntDP"],
    "Inference Time (ms)": [0.2, 5.7, 4.6, 5.2, 0.11, 0.12, 0.11, 0.11, 0.12, 0.12, 0.14, 4.0],
    "Performance": [8085.3, 6425.1, 7142.6, 5665.9, 5756.2, 5762.4, 4583.8, 5030.9, 2781.9, 2100.2, 7000.0, 6000.0]
}
df = pd.DataFrame(data)

# 计算动作频率 (kHz) = 1 / Inference Time (ms)
df['Action Frequency (kHz)'] = 1 / df['Inference Time (ms)']

# 2. 设置绘图风格 (模仿论文风格)
plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.2

fig, ax = plt.subplots(figsize=(8, 4))

# 3. 定义每个方法的样式 (形状、颜色、大小)
# 尽量复刻上传图片的配色方案
styles = {
    "SMFP(Ours)": {"marker": "*", "color": "red", "s": 500, "zorder": 10},  # 红色五角星，最大，最上层
    "QVPO":       {"marker": "s", "color": "blue", "s": 150, "zorder": 5},  # 蓝色方块
    "DIME":       {"marker": "D", "color": "green", "s": 150, "zorder": 5}, # 绿色菱形
    "DIPO":       {"marker": "h", "color": "cyan", "s": 150, "zorder": 5}, # 青色六边形
    "FPMD-R":       {"marker": "^", "color": "yellow", "s": 150, "zorder": 5},# 黄色上三角
    "FPMD-M":       {"marker": "p", "color": "black", "s": 150, "zorder": 5},  # 黑色五角形
    "TD3":        {"marker": "X", "color": "gray", "s": 150, "zorder": 5},  # 灰色叉号
    "SAC":        {"marker": "^", "color": "purple", "s": 150, "zorder": 5},# 紫色上三角
    "PPO":        {"marker": "D", "color": "darkgreen", "s": 150, "zorder": 5}, # 深绿菱形
    "SPO":        {"marker": "v", "color": "orange", "s": 150, "zorder": 5}, # 橙色下三角
    "Crossq":     {"marker": "P", "color": "gold", "s": 150, "zorder": 5},   # 金色加号
    "MaxEntDP":   {"marker": "8", "color": "brown", "s": 150, "zorder": 5},  # 棕色八边形
}

# 4. 循环绘制每个点
for i, row in df.iterrows():
    method = row['Method']
    style = styles.get(method, {"marker": "o", "color": "black", "s": 100})
    
    ax.scatter(row['Action Frequency (kHz)'], row['Performance'], 
               c=style['color'], marker=style['marker'], s=style['s'], 
               edgecolors='black', linewidth=1.0, # 黑色描边
               label=method, zorder=style.get('zorder', 1))

    # 专门给 Ours 加文字标注
    if "Ours" in method:
        ax.text(row['Action Frequency (kHz)'], row['Performance'] - 500, "Ours", 
                fontsize=17, fontweight='bold', ha='center', va='top', zorder=12)

# 5. 设置坐标轴与网格
ax.grid(True, linestyle='--', alpha=0.5, color='lightgray')
ax.set_xlabel("Action Frequency (kHz)", fontsize=17, fontweight='bold')
ax.set_ylabel("Episodic Reward", fontsize=17, labelpad=5, fontweight='bold')
plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), fontweight="bold")

# 设置坐标范围 (根据数据微调，留出空间)
# 频率最大值约 9.1 kHz (1/0.11)，设置到 10.0 比较合适
ax.set_xlim(-1, 10.0)  # X轴范围
ax.set_ylim(1500, 8500) # Y轴范围

# 6. 图例设置 (放在底部，分列显示)
handles, labels = ax.get_legend_handles_labels()
# 可以根据需要调整图例顺序，这里保持默认
ax.legend(handles, labels, loc='lower left', ncol=3, fontsize=12, 
          frameon=True, edgecolor='lightgray', borderpad=1.2, markerscale=0.8)

plt.tight_layout()
plt.savefig('./dataset/teaser_performance.pdf', dpi=300, transparent=True)
plt.show()