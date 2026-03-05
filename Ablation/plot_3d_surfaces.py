import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import axes3d
import os

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude):
    return amplitude * np.exp(-(((x - x0)**2)/(2 * sigma_x**2) + ((y - y0)**2)/(2 * sigma_y**2)))

def plot_surface(mode='single', filename='surface.png'):
    # Grid setup
    # Adjust range and resolution to match the look
    x = np.linspace(-3, 3, 25) 
    y = np.linspace(-3, 3, 25)
    X, Y = np.meshgrid(x, y)
    
    if mode == 'single':
        # Single Gaussian (Unimodal)
        # Centered, symmetric
        Z = gaussian_2d(X, Y, 0, 0, 1.2, 1.2, 1.5)
    else:
        # Multi-modal
        # Trying to replicate the 3-peak structure in the image
        # One main peak in the back center
        Z = gaussian_2d(X, Y, 0, 1.0, 0.8, 0.8, 1.2)
        # Two side peaks
        Z += gaussian_2d(X, Y, -1.5, -0.5, 0.6, 0.6, 0.8)
        Z += gaussian_2d(X, Y, 1.5, -0.5, 0.6, 0.6, 0.8)
        # A smaller one in front center
        Z += gaussian_2d(X, Y, 0, -1.5, 0.5, 0.5, 0.4)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Custom colormap: truncated viridis to make bottom lighter (remove dark purple)
    # Start from 0.2 to skip the darkest purple/blue
    try:
        cmap = plt.get_cmap('viridis')
    except AttributeError:
        # For newer matplotlib versions
        cmap = plt.cm.viridis
        
    new_colors = cmap(np.linspace(0.2, 1.0, 256))
    light_viridis = mcolors.ListedColormap(new_colors)

    # Plot surface with custom colormap
    # No contours: linewidth=0, antialiased=False
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=light_viridis, 
                   linewidth=0, antialiased=False, shade=True, alpha=1.0)
    
    # Remove axes, ticks, labels for clean look
    ax.set_axis_off()
    
    # Set transparent background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Adjust view angle to match the images
    # Image seems to be viewed from slightly above
    ax.view_init(elev=35, azim=-60)
    
    # Tight layout to minimize whitespace
    plt.tight_layout()
    
    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, filename)
    plt.savefig(output_path, transparent=True, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_surface('single', 'unimodal_surface.svg')
    plot_surface('multi', 'multimodal_surface.svg')
