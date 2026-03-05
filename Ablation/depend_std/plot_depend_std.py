import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

def plot_depend_std():
    # Set the style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 26
    plt.rcParams['axes.linewidth'] = 2.0
    
    # Create figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files = [
        ('Hopper-v4', 'Hopper.csv'),
        ('Walker2d-v4', 'Walker.csv')
    ]

    # Styles
    # We map 'true' to the Blue style (primary) and 'false' to the Orange style (secondary)
    # matching the visual hierarchy of plot_q_agg.py
    styles = {
        'true': {'color': '#1f77b4', 'label': 'SMFP', 'linestyle': '-', 'marker': 'o'}, # Blue
        'false': {'color': '#ff7f0e', 'label': 'W/o SAC Loss', 'linestyle': '--', 'marker': 's'} # Orange
    }

    def smooth_data(series, window_size=10):
        return series.rolling(window=window_size, min_periods=1, center=True).mean()

    # Collect handles for global legend
    global_handles = {}

    for idx, (env_name, csv_file) in enumerate(files):
        ax = axes[idx]
        file_path = os.path.join(script_dir, csv_file)
        
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Normalize Step to 0-1 (assuming max is 1e6)
        if 'Step' in df.columns:
            df['Step'] = df['Step'] / 1000000.0
        
        # Identify columns
        data_cols = {}
        for col in df.columns:
            if 'evaluation/episode.return' in col and '__' not in col:
                if 'agent.state_dependent_std: true' in col:
                    data_cols['true'] = col
                elif 'agent.state_dependent_std: false' in col:
                    data_cols['false'] = col
        
        # Plot
        # Order: True then False (similar to Mean then Min in reference)
        for key in ['true', 'false']: 
            if key not in data_cols:
                continue
                
            main_col = data_cols[key]
            min_col = main_col + '__MIN'
            max_col = main_col + '__MAX'
            
            style = styles[key]
            
            x_data = df['Step']
            y_main = df[main_col]
            y_min_data = df[min_col] if min_col in df.columns else y_main
            y_max_data = df[max_col] if max_col in df.columns else y_main
            
            # Reduce shadow size for Hopper-v4
            if env_name == 'Hopper-v4':
                y_min_data = y_main - (y_main - y_min_data) * 0.5
                y_max_data = y_main + (y_max_data - y_main) * 0.5
            
            # Apply smoothing
            # Assuming 1e6 steps total, use a window that covers ~5% of data for visual smoothness
            # If x_data is just indices or step counts, we can estimate window size
            window_size = max(1, int(len(x_data) * 0.05))
            y_main = smooth_data(y_main, window_size)
            y_min_data = smooth_data(y_min_data, window_size)
            y_max_data = smooth_data(y_max_data, window_size)

            # Plot Line
            line, = ax.plot(x_data, y_main, color=style['color'], linewidth=4.0, 
                           linestyle=style['linestyle'], label=style['label'],
                           marker=style['marker'], markersize=12, markerfacecolor='white', 
                           markeredgecolor=style['color'], markeredgewidth=3.0, markevery=0.1)
            
            if key not in global_handles:
                global_handles[key] = line

            # Fill between
            ax.fill_between(x_data, y_min_data, y_max_data, color=style['color'], alpha=0.15, linewidth=0)

        # Configure axes
        ax.set_xlabel('Steps', fontsize=32, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Episodic Reward', fontsize=32, fontweight='bold')
        
        ax.set_title(env_name, fontsize=34, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)
        
        # Y-axis formatter (k units)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '0' if x == 0 else f'{x/1000:g}k'))
        
        # X-axis ticks
        ax.set_xticks([0.0, 0.5, 1.0])
        
        # Add 1e6 label
        ax.text(1.0, -0.12, '1e6', transform=ax.transAxes, 
                ha='right', va='top', fontsize=28, fontweight='bold')

        # Make tick labels bold and larger
        ax.tick_params(axis='both', which='major', labelsize=26)
        for label_obj in ax.get_xticklabels() + ax.get_yticklabels():
            label_obj.set_fontweight('bold')
        
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Adjust Y-limits slightly to show 0 clearly if needed
        y_min_val, y_max_val = ax.get_ylim()
        if y_min_val > -0.02 * y_max_val:
             ax.set_ylim(bottom=-0.02 * y_max_val)

    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.98, wspace=0.12, top=0.85, bottom=0.25)
    
    # Unified Legend at bottom
    font_prop = {'weight': 'bold', 'size': 28}
    
    # Order: True, False
    final_handles = []
    final_labels = []
    for k in ['true', 'false']:
        if k in global_handles:
            final_handles.append(global_handles[k])
            final_labels.append(styles[k]['label'])

    fig.legend(final_handles, final_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(final_labels), prop=font_prop, frameon=False, handlelength=2.0, columnspacing=2.0)
    
    save_path = os.path.join(script_dir, 'depend_std_ablation.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_depend_std()
