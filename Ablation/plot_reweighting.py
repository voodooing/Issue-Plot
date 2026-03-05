import matplotlib.pyplot as plt
import numpy as np
import os

def plot_reweighting_comparison():
    # ===================== 核心样式优化（学术顶刊风格）=====================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 20,          # 整体字体下调，避免拥挤
        'axes.linewidth': 1.0,    # 边框线条更细腻
        'text.color': '#000000',
        'axes.labelcolor': '#000000',
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'axes.edgecolor': '#000000',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'grid.alpha': 0.3,        # 网格线半透明，不抢焦点
        'grid.linewidth': 0.8,
    })

    # ===================== 数据准备（不变）=====================
    envs = ['Hopper', 'Walker2d', 'Ant', 'HalfCheetah', 'Humanoid', 'HumanoidStandup', 'Swimmer']
    reweighting_means = np.array([3645.47, 7099.91, 8249.79, 15624.83, 12390.45, 273183.56, 155.89])
    no_reweighting_means = np.array([3387.93, 6408.62, 7939.60, 12567.09, 8634.45, 270893.89, 145.35])
    
    # 计算百分比贡献
    total = reweighting_means + no_reweighting_means
    pct_no = (no_reweighting_means / total) * 100
    pct_rw = (reweighting_means / total) * 100
    
    # ===================== 绘图设置（核心优化）=====================
    fig, ax = plt.subplots(figsize=(12, 6))  # 加宽加高，避免拥挤
    y_pos = np.arange(len(envs))
    height = 0.7  # 减小柱子高度，增加间距（原0.7→0.5）
    gap = 0.1     # 环境间额外间距（可选，这里靠height控制）

    # 🔥 优化1：学术级柔和配色（替换刺眼的红+蓝）
    color_nr = '#E0E0E0'  # 浅灰色（No Reweighting，基线）
    color_rw = '#4472C4'  # 深蓝色（With Reweighting，Ours），顶刊常用'#4472C4'

    # 绘制水平堆叠柱状图 - 交换顺序：蓝色(Ours)在左，灰色(No)在右
    
    # 绘制 With Reweighting (Ours) - 现在在左边 (作为基底)
    bars_rw = ax.barh(y_pos, pct_rw, height, 
                      color=color_rw, label='SMFP (Ours)', 
                      edgecolor='white', hatch='///', linewidth=0.0)
    
    # 绘制 No Reweighting - 现在在右边 (叠加在 rw 之上)
    bars_no = ax.barh(y_pos, pct_no, height, left=pct_rw, 
                      color=color_nr, label='W/o Mirror Descent', 
                      edgecolor='black', linewidth=0.0)

    # 🔥 优化2：50%分界线更细腻（原粗线→细线+浅灰，不抢焦点）
    ax.axvline(50, color='#666666', linestyle='--', linewidth=1.2, zorder=3)
    
    # 🔥 优化3：百分比标签统一样式（无大小差异，适配背景色）
    for i, (p_no, p_rw) in enumerate(zip(pct_no, pct_rw)):
        # With Reweighting 标签（改为白底黑字带框） - 现在在左边 (25% 处)
        ax.text(25, i, f"{p_rw:.1f}%", ha='center', va='center', 
                color='black', fontweight='bold', fontsize=16,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.0))
        
        # No Reweighting 标签（浅灰背景用黑色，加粗但字体更小） - 现在在右边 (75% 处)
        ax.text(75, i, f"{p_no:.1f}%", ha='center', va='center', 
                color='black', fontweight='normal', fontsize=16)

    # ===================== 轴域优化（核心美观度）=====================
    # Y轴：反转+右对齐，标签字体优化
    ax.set_yticks(y_pos)
    ax.set_yticklabels(envs, fontsize=17, fontweight='bold')
    ax.invert_yaxis()
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.tick_params(axis='y', pad=8)  # Y轴标签和柱子拉开间距

    # X轴：添加浅灰网格线，刻度更清晰
    ax.set_xlim(0, 100)
    ax.set_xlabel('Relative Performance (%)', fontsize=18, fontweight='bold', labelpad=10)
    ax.tick_params(axis='x', labelsize=15, pad=5)
    ax.xaxis.grid(True, linestyle='-', color='#F0F0F0')  # 浅灰网格线
    ax.set_axisbelow(True)  # 网格线在柱子下方

    # 🔥 优化4：边框精简（只保留底部+右侧Y轴，更清爽）
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # 🔥 优化5：图例优化（位置更合理，无框，字体统一，加粗）
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, 
              frameon=False, prop={'weight': 'bold', 'size': 20}, columnspacing=2.0)

    # 🔥 优化6：布局调整（避免元素挤压）
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # 预留图例空间

    # 保存图片（优化DPI+矢量图清晰度）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'reweighting_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Optimized plot saved to {output_path}")

if __name__ == "__main__":
    plot_reweighting_comparison()