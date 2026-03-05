import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from matplotlib.ticker import FuncFormatter

def plot_q_agg():
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
    styles = {
        'min': {'color': '#1f77b4', 'label': 'Min', 'linestyle': '-', 'marker': 'o'}, # Blue
        'mean':  {'color': '#ff7f0e', 'label': 'Mean',  'linestyle': '--', 'marker': 's'}  # Orange
    }

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
        
        # Identify columns
        # We look for "agent.q_agg: min" and "agent.q_agg: mean"
        data_cols = {}
        for col in df.columns:
            if 'evaluation/episode.return' in col and '__' not in col:
                if 'agent.q_agg: min' in col:
                    data_cols['min'] = col
                elif 'agent.q_agg: mean' in col:
                    data_cols['mean'] = col
        
        # Plot
        for key in ['mean', 'min']: # Order: Mean then Min
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
        
        # Adjust Y-limits slightly to show 0 clearly if needed, similar to alpha plot
        y_min_val, y_max_val = ax.get_ylim()
        if y_min_val > -0.02 * y_max_val:
             ax.set_ylim(bottom=-0.02 * y_max_val)


    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.98, wspace=0.12, top=0.85, bottom=0.25)
    
    # Unified Legend at bottom
    font_prop = {'weight': 'bold', 'size': 28}
    
    # Order: Mean, Min
    final_handles = []
    final_labels = []
    for k in ['mean', 'min']:
        if k in global_handles:
            final_handles.append(global_handles[k])
            final_labels.append(styles[k]['label'])

    fig.legend(final_handles, final_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(final_labels), prop=font_prop, frameon=False, handlelength=2.0, columnspacing=2.0)
    
    save_path = os.path.join(script_dir, 'q_agg_ablation.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_q_agg()
