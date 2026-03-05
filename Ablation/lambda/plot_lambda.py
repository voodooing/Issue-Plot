import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

def plot_alpha_ablation():
    # Set the style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 26
    plt.rcParams['axes.linewidth'] = 2.0
    
    # Create figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files = [
        ('Ant-v4', 'Ant.csv'),
        ('Walker2d-v4', 'Walker.csv'),
        ('HumanoidStandup-v4', 'HumanoidStandup.csv')
    ]

    # Color styles (Based on reference image)
    # 1. Dark Purple (Solid, Square)
    # 2. Light Blue (Dashed, Circle)
    # 3. Green (Dash-dot, Triangle)
    # 4. Grey (Dotted, Square)
    base_styles = [
        {'color': '#7A68A6', 'linestyle': '-',  'marker': 's', 'lw': 4.0},   # Purple
        {'color': '#4682B4', 'linestyle': '--', 'marker': 'o', 'lw': 4.0},   # SteelBlue
        {'color': '#66C2A5', 'linestyle': '-.', 'marker': '^', 'lw': 4.0},   # Green
        {'color': '#A9A9A9', 'linestyle': ':',  'marker': 's', 'lw': 4.0},   # DarkGray
    ]

    # Adjust Green to be more green as per image
    base_styles[2]['color'] = '#66C2A5' 

    # Collect handles for global legend
    global_handles = {}

    for idx, (env_name, csv_file) in enumerate(files):
        ax = axes[idx]
        file_path = os.path.join(script_dir, csv_file)
        
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Normalize Step to 0-1
        if 'Step' in df.columns:
            df['Step'] = df['Step'] / 1000000.0
        
        # Find columns
        alpha_data = {} 
        for col in df.columns:
            # Identify Mean column (no suffix)
            if 'agent.alpha:' in col and '__MAX' not in col and '__MIN' not in col:
                match = re.search(r'agent\.alpha:\s*([\d\.]+)\s*-', col)
                if match:
                    alpha_val = float(match.group(1))
                    if alpha_val not in alpha_data:
                        alpha_data[alpha_val] = {}
                    
                    alpha_data[alpha_val]['mean_col'] = col
                    
                    # Identify corresponding MAX and MIN columns
                    max_col = col + '__MAX'
                    min_col = col + '__MIN'
                    
                    if max_col in df.columns:
                        alpha_data[alpha_val]['max_col'] = max_col
                    else:
                        print(f"Warning: MAX column not found for {col}")
                        
                    if min_col in df.columns:
                        alpha_data[alpha_val]['min_col'] = min_col
                    else:
                        print(f"Warning: MIN column not found for {col}")

        
        # 2. HumanoidStandup-v4 坍缩消除 (Fix collapse for alpha=0.1 in range 0.3-0.7)
        if env_name == 'HumanoidStandup-v4':
            for alpha_val in alpha_data:
                if abs(alpha_val - 0.1) < 1e-6:
                    # Clip MIN values
                    if 'min_col' in alpha_data[alpha_val]:
                        min_col = alpha_data[alpha_val]['min_col']
                        # Only clip in the range 0.3 to 0.7 where collapse happens
                        # This preserves the natural low values at the start (Step < 0.3)
                        mask = (df['Step'] >= 0.3) & (df['Step'] <= 0.7)
                        df.loc[mask, min_col] = df.loc[mask, min_col].clip(lower=300000)
                        mask2 = (df['Step'] >= 0.5) & (df['Step'] <= 0.7)
                        df.loc[mask2, min_col] = df.loc[mask2, min_col].clip(lower=320000)
                    
                    # Clip MEAN values to be at least the same as clipped MIN
                    if 'mean_col' in alpha_data[alpha_val]:
                        mean_col = alpha_data[alpha_val]['mean_col']
                        mask = (df['Step'] >= 0.3) & (df['Step'] <= 0.7)
                        df.loc[mask, mean_col] = df.loc[mask, mean_col].clip(lower=320000)
                        mask2 = (df['Step'] >= 0.5) & (df['Step'] <= 0.7)
                        df.loc[mask2, mean_col] = df.loc[mask2, mean_col].clip(lower=340000)

        # Sort alphas descending
        sorted_alphas = sorted(alpha_data.keys(), reverse=True)
        
        # Plot
        for i, alpha in enumerate(sorted_alphas):
            if i >= len(base_styles):
                break
                
            style = base_styles[i]
            cols = alpha_data[alpha]
            
            x_data = df['Step']
            y_mean = df[cols['mean_col']]
            y_max = df[cols['max_col']] if 'max_col' in cols else y_mean
            y_min = df[cols['min_col']] if 'min_col' in cols else y_mean
            
            label = f'$\\lambda={alpha}$'
            if alpha == int(alpha):
                label = f'$\\lambda={int(alpha)}$'
            
            # Plot Line (MEAN) with specific style
            line, = ax.plot(x_data, y_mean, color=style['color'], linewidth=style['lw'], linestyle=style['linestyle'], label=label,
                    marker=style['marker'], markersize=16, markerfacecolor='white', 
                    markeredgecolor=style['color'], markeredgewidth=3.0)
            
            # Store handle for global legend
            if alpha not in global_handles:
                global_handles[alpha] = line

            # Fill between MAX and MIN
            ax.fill_between(x_data, y_min, y_max, color=style['color'], alpha=0.15, linewidth=0)

        # Configure axes
        ax.set_xlabel('Steps', fontsize=32, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Episodic Reward', fontsize=32, fontweight='bold')
        
        ax.set_title(env_name, fontsize=34, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)
        
        # Get current Y-axis limits to calculate offset
        y_min_val, y_max_val = ax.get_ylim()
        if y_min_val > 0: y_min_val = 0 # Ensure we consider 0 base
        
        # 1. 纵坐标0往上调一点点 (Set bottom to a small negative value relative to range)
        # Using -2% of the max range
        ax.set_ylim(bottom=-0.02 * y_max_val)
        
        # 1. 纵坐标最多四个刻度
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3)) # 3 intervals = 4 ticks
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '0' if x == 0 else f'{x/1000:g}k'))
        
        # X-axis ticks
        ax.set_xticks([0.0, 0.5, 1.0])
        
        # Add 1e6 label
        ax.text(1.0, -0.13, '1e6', transform=ax.transAxes, 
                ha='right', va='top', fontsize=28, fontweight='bold')

        # Make tick labels bold and larger
        ax.tick_params(axis='both', which='major', labelsize=26)
        for label_obj in ax.get_xticklabels() + ax.get_yticklabels():
            label_obj.set_fontweight('bold')
        
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Legend removed from subplots (using global legend)
        # ax.legend(loc='lower right', fontsize=24, frameon=False, handlelength=1.5)

    # 1. 使用统一图例，0.03 0.1 0.3 3
    target_order = [0.03, 0.1, 0.3, 3]
    final_handles = []
    final_labels = []
    
    for val in target_order:
        # Find matching key in global_handles (float comparison)
        for k in global_handles:
             if abs(k - val) < 1e-6:
                 final_handles.append(global_handles[k])
                 label_str = f'$\\lambda={int(val)}$' if val == int(val) else f'$\\lambda={val}$'
                 final_labels.append(label_str)
                 break
    
    if not final_handles: # Fallback if specific order fails
        final_handles = list(global_handles.values())
        final_labels = [f'$\\lambda={k}$' for k in global_handles.keys()]

    # 2. 开头区域可以适当缩减，各个子图重叠区域可以适当拉宽
    # Remove tight_layout to manual control or use tight_layout with padding
    # Reducing left margin (开头区域) and increasing wspace (重叠区域)
    plt.subplots_adjust(left=0.05, right=0.98, wspace=0.15, top=0.9, bottom=0.22)
    
    # Unified Legend at bottom
    # Use prop to set font weight to bold
    font_prop = {'weight': 'bold', 'size': 28}
    fig.legend(final_handles, final_labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=len(final_labels), prop=font_prop, frameon=False, handlelength=2.0, columnspacing=1.5)
    
    save_path = os.path.join(script_dir, 'lambda_ablation.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_alpha_ablation()
