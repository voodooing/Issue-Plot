import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches # 引入补丁库，用于创建透明句柄

def plot_on_ax(ax, file_path, env_name, y_ticks=None, y_lim=None, show_ylabel=True):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Load data
    df = pd.read_csv(file_path)
    
    # Downsample by half
    df = df.iloc[::1].reset_index(drop=True)
    
    # Check columns
    cols = [c for c in df.columns if c != 'Step']
    
    # Parse std values from column names
    col_std_map = {}
    for c in cols:
        try:
            val = float(c.split(':')[-1].strip())
            col_std_map[c] = val
        except:
            col_std_map[c] = 0
            
    # Sort columns by std value (Large -> Small)
    sorted_cols = sorted(cols, key=lambda x: col_std_map[x], reverse=True)
    
    # ================= 修改 1: 标签只保留数值占位符 =================
    styles = [
        {'color': '#021f4b', 'label': '1', 'marker': 'o'},   # Darkest Blue
        {'color': '#1860aa', 'label': '3', 'marker': 's'},   # Dark Blue
        {'color': '#539ecd', 'label': '5', 'marker': '^'},   # Medium Blue
        {'color': '#99d3f5', 'label': '7', 'marker': 'D'},   # Light Blue
    ]
    # =============================================================
    
    # X-axis: Step 0-1
    x_data = df['Step']
    
    # Plot each line
    for i, col in enumerate(sorted_cols):
        if i >= len(styles):
            break
            
        std_val = col_std_map[col]
        style = styles[i]
        
        y_data = df[col]
        
        # Smooth the data
        total_steps = 1e6
        if len(x_data) > 1:
            step_diff = (x_data.iloc[1] - x_data.iloc[0]) * total_steps
            window_steps = 5000
            window_size = int(window_steps / step_diff)
            if window_size < 1: window_size = 1
            
            # --- Aggressive Outlier Removal ---
            stable_window_steps = 50000 
            stable_window_size = int(stable_window_steps / step_diff)
            if stable_window_size < 1: stable_window_size = 1
            
            y_trend = y_data.rolling(window=stable_window_size, min_periods=1, center=True).quantile(0.90)
            
            threshold_ratio = 0.90
            y_filtered = y_data.copy()
            
            mask_outliers = (y_filtered < (y_trend * threshold_ratio)) & (y_trend > 0) & (x_data > 0.01)
            y_filtered[mask_outliers] = np.nan
            
            y_interpolated = y_filtered.interpolate(method='linear', limit_direction='both')
            y_interpolated = y_interpolated.fillna(y_data)

            # Final Smoothing
            y_smoothed = y_interpolated.rolling(window=window_size, min_periods=1, center=True).mean()
            # 大大增加 std (x20)
            std_multiplier = 8.0 if env_name == 'Humanoid-v4' else 5.0
            y_std = y_interpolated.rolling(window=window_size, min_periods=1, center=True).std().fillna(0) * std_multiplier
        else:
            y_smoothed = y_data
            y_std = np.zeros_like(y_data)
        
        # Label with actual value
        label = style['label'].format(std_val)
        
        # Sample at specific steps
        target_x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # Interpolate to get values at target steps
        sampled_y = np.interp(target_x, x_data, y_smoothed)
        sampled_std = np.interp(target_x, x_data, y_std)
        
        # Plot line using sampled data
        ax.plot(target_x, sampled_y, color=style['color'], linewidth=2.5, label=label, 
                marker=style['marker'], markersize=10, markerfacecolor='white', markeredgecolor=style['color'], markeredgewidth=2.5)
        ax.fill_between(target_x, sampled_y - sampled_std, sampled_y + sampled_std, color=style['color'], alpha=0.15, linewidth=0)

    # Configure axes
    ax.set_xlabel('Steps', fontsize=28, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Episodic Reward', fontsize=28, fontweight='bold')
    ax.set_title(env_name, fontsize=30, fontweight='bold', pad=15)
    
    ax.set_xlim(0, 1.0)
    if y_lim:
        ax.set_ylim(y_lim)
    if y_ticks:
        ax.set_yticks(y_ticks)
    
    # X-axis ticks
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Add 1e6 label
    ax.text(1.04, -0.12, '1e6', transform=ax.transAxes, 
            ha='center', va='top', fontsize=24, fontweight='bold')

    # Make tick labels bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    ax.grid(True, linestyle='--', alpha=0.3)

def plot_ablation():
    # Set the style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 26
    plt.rcParams['axes.linewidth'] = 1.5
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. HalfCheetah-v4 (Left)
    file_path_hc = os.path.join(script_dir, 'HalfCheetah-v4_noise.csv')
    plot_on_ax(axes[0], file_path_hc, 'HalfCheetah-v4', 
               y_ticks=[0, 8000, 16000], 
               y_lim=[-500, 18000])
               
    # 2. Humanoid-v4 (Right)
    file_path_hu = os.path.join(script_dir, 'Humanoid-v4_noise.csv')
    plot_on_ax(axes[1], file_path_hu, 'Humanoid-v4', 
               y_ticks=[0, 6000, 12000], 
               y_lim=[-500, 14000],
               show_ylabel=False)

    # ================= 修改 2: 构建自定义图例 =================
    
    # 获取原本的句柄和标签 (只含数值)
    handles, labels = axes[0].get_legend_handles_labels()
    
    # 创建一个不可见(透明)的 Patch，仅用于显示标题文字
    # alpha=0 让方块完全透明，但保留占位功能
    # 使用 LaTeX \mathbf 只对非希腊字母部分加粗，避免 \sigma 变得过粗
    title_str = r'$-\mathcal{H}_{\mathrm{target}} \mathbf{:}$'
    title_handle = mpatches.Patch(color='none', label=title_str)
    
    # 将标题句柄插入到列表的最前面
    handles.insert(0, title_handle)
    labels.insert(0, title_str)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.subplots_adjust(wspace=0.15)
    
    # 创建全局图例
    legend = fig.legend(handles, labels, 
               loc='lower center',         
               bbox_to_anchor=(0.5, 0.06), 
               ncol=len(labels),           # 列数 = 标题 + 数值个数
               frameon=False,              
               fontsize=24,
               columnspacing=0.8,          # 控制列与列之间的间距
               handletextpad=0.2,          # 控制图标和文字之间的间距（对标题来说是透明图标和文字的间距）
               handlelength=2.0)           # 图标长度
    
    # 设置图例字体加粗 (跳过第一个标题元素，防止 sigma 再次被加粗)
    for i, text in enumerate(legend.get_texts()):
        if i == 0:
            continue
        text.set_fontweight('bold')

    # Save
    save_path = os.path.join(script_dir, 'noise_ablation.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_ablation()