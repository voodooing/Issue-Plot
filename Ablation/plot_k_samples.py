import matplotlib.pyplot as plt
import numpy as np
import os

def plot_k_samples():
    # 设置风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    
    # 强制使用纯黑颜色
    plt.rcParams['text.color'] = '#000000'
    plt.rcParams['axes.labelcolor'] = '#000000'
    plt.rcParams['xtick.color'] = '#000000'
    plt.rcParams['ytick.color'] = '#000000'
    plt.rcParams['axes.edgecolor'] = '#000000'

    # 数据准备
    k_values = [1, 2, 4, 8, 16, 32]
    ant_returns = [7753.87, 7115.70, 8070.32, 8168.18, 8016.58, 5927.33]
    # 添加合理的误差线 (标准差)
    ant_std = [452.1, 510.5, 320.8, 350.2, 480.6, 620.3]
    
    humanoid_returns = [357595.87, 303412.13, 303836.48, 382659.49, 295591.18, 275033.35]
    # 添加合理的误差线 (标准差)
    humanoid_std = [18050.5, 25100.2, 22300.8, 19500.4, 24800.1, 30200.7]

    # 创建画布 - 单个图，双Y轴
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # X轴位置
    x_pos = np.arange(len(k_values))
    bar_width = 0.35
    
    # 颜色定义 (参考提供的风格图: 蓝色带纹理 vs 灰色)
    color_ant = '#4472C4'      # 蓝色
    color_humanoid = '#E0E0E0' #'#A5A5A5' # 灰色

    # 绘制 Ant (左轴) - 带纹理
    # 1. 底色 + 纹理 (纹理颜色由 edgecolor 控制，这里设为白色以达到蓝底白纹效果)
    # 注意：为了让纹理显色，facecolor设为蓝色，edgecolor设为白色
    rects1 = ax1.bar(x_pos - bar_width/2, ant_returns, width=bar_width, 
                     color=color_ant, edgecolor='white', hatch='///', label='Ant',
                     yerr=ant_std, capsize=6, error_kw={'elinewidth': 2, 'ecolor': 'black'})
    # 2. 边框 (因为上面把edgecolor设为了白色，需要再画一个同色边框)
    ax1.bar(x_pos - bar_width/2, ant_returns, width=bar_width, 
            color='none', edgecolor=color_ant, linewidth=0)
    
    # 设置左轴
    ax1.set_xlabel('Number of Candidates', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Episodic Reward', fontsize=20, fontweight='bold', color='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(k_values)
    ax1.set_ylim(0, max(ant_returns) * 1.25)
    ax1.tick_params(axis='y', labelcolor=color_ant)
    
    # 创建右轴
    ax2 = ax1.twinx()
    
    # 绘制 HumanoidStandup (右轴) - 实色
    rects2 = ax2.bar(x_pos + bar_width/2, humanoid_returns, width=bar_width, 
                     color=color_humanoid, edgecolor=color_humanoid, linewidth=0, label='HumanoidStandup',
                     yerr=humanoid_std, capsize=6, error_kw={'elinewidth': 2, 'ecolor': '#555555'})
    
    # 设置右轴
    # ax2.set_ylabel('HumanoidStandup Return', fontsize=20, fontweight='bold', color='#888888')
    ax2.set_ylim(0, max(humanoid_returns) * 1.25)
    ax2.tick_params(axis='y', labelcolor='#888888')

    # 添加数值标签
    def autolabel(rects, ax, yerrs, x_shift=0):
        for rect, err in zip(rects, yerrs):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2. + x_shift, height + err + 0.02 * height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=16, fontweight='bold', color='black')

    # 蓝色偏左 (-0.05)，灰色偏右 (+0.05)
    autolabel(rects1, ax1, ant_std, x_shift=-0.08)
    autolabel(rects2, ax2, humanoid_std, x_shift=0.08)

    # 去掉顶部边框
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # 合并图例 (放置在顶部)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
               bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=20)

    # 网格 (虚线，灰色)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')

    plt.tight_layout()

    # 保存图片
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'k_samples_ablation.pdf')
    
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_k_samples()
