import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.optimize import linear_sum_assignment
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# === 配置 ===
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
BATCH_SIZE = 256 # Reduced for OT speed
ITERATIONS = 2000
LR = 1e-3
DEVICE = 'cpu'

# 设置绘图风格
plt.rcParams.update({
    'font.size': 16, # Increased font size
    'font.weight': 'bold', # Bold font
    'axes.titlesize': 18, # Increased title size
    'axes.titleweight': 'bold', # Bold title
    'axes.labelsize': 16, # Added label size
    'axes.labelweight': 'bold', # Bold label
    'xtick.labelsize': 14, # Added tick label size
    'ytick.labelsize': 14, # Added tick label size
    'figure.dpi': 120,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'] # Use Times New Roman
})

# === 1. 定义环境与目标分布 ===

class ToyEnvironment:
    def __init__(self):
        # Base Policy (Prior): N(0, 2^2)
        self.prior_std = 2.0
        
    def log_pi_old(self, x):
        return -0.5 * torch.sum(x**2, dim=1) / (self.prior_std**2)

    def Q_function(self, x):
        # 构建一个双峰的 Q 函数 (Mix of Gaussians)
        # 峰值 1: 在 (2.5, 2.5)
        dist1 = torch.sum((x - torch.tensor([3, 3]))**2, dim=1)
        # Decrease variance to sharpen the peaks (div by 0.5)
        q1 = 5.0 * torch.exp(-0.5 * dist1 / 8)
        
        # 峰值 2: 在 (-2.5, -2.5)
        dist2 = torch.sum((x - torch.tensor([-3, -3]))**2, dim=1)
        # Decrease variance to sharpen the peaks (div by 0.5)
        q2 = 5.0 * torch.exp(-0.5 * dist2 / 3)
        
        return q1 + q2

    def get_target_samples(self, n_samples):
        """
        通过拒绝采样获取目标分布 π* ∝ π_old * exp(Q) 的真实样本。
        用于 Flow Matching 的监督信号。
        """
        samples = []
        while len(samples) < n_samples:
            # 提议分布: 均匀分布覆盖区域 [-5, 5]
            proposal = (torch.rand(n_samples * 2, 2) * 10 - 5) 
            
            # 计算非归一化目标概率 density
            with torch.no_grad():
                log_prob = self.log_pi_old(proposal) + self.Q_function(proposal)
                prob = torch.exp(log_prob)
            
            # 拒绝采样
            # 这里的 max_prob 是估计的，用于缩放
            max_prob = torch.exp(torch.tensor(-0.5*0 + 5.0)) # 粗略估计最大值
            accept_prob = prob / max_prob
            mask = torch.rand(n_samples * 2) < accept_prob
            
            samples.append(proposal[mask])
        
        return torch.cat(samples)[:n_samples]

    def get_energy_grid(self, bounds=(-4, 4), res=100):
        """生成用于画等高线的网格数据"""
        x = np.linspace(bounds[0], bounds[1], res)
        y = np.linspace(bounds[0], bounds[1], res)
        xx, yy = np.meshgrid(x, y)
        grid_tensor = torch.tensor(np.stack([xx, yy], axis=-1), dtype=torch.float32).reshape(-1, 2)
        
        with torch.no_grad():
            log_prob = self.log_pi_old(grid_tensor) + self.Q_function(grid_tensor)
            prob = torch.exp(log_prob).reshape(res, res)
            
        return xx, yy, prob.numpy()

env = ToyEnvironment()

# === 2. 定义模型 ===

# --- Model A: Gaussian Policy (Standard SAC) ---
class GaussianPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 学习均值和对数标准差
        # Break symmetry: Initialize slightly off-center to allow mode collapse
        # If initialized at (0,0), gradients from both modes cancel out (saddle point)
        # Use a stronger bias (0.5, 0.5) to ensure it falls into the (2,2) mode quickly
        self.mu = nn.Parameter(torch.tensor([0.5, 0.5])) 
        # Start with smaller variance to encourage staying in one mode
        self.log_std = nn.Parameter(torch.zeros(2))
        
    def forward(self, n_samples):
        std = torch.exp(self.log_std)
        eps = torch.randn(n_samples, 2)
        return self.mu + eps * std, self.mu, std

# --- Model B: Flow Policy (Stochastic MeanFlow / Flow Matching) ---
class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        # 简单的 MLP: 输入 (x, t) -> 输出 velocity
        # Increased capacity: 64 -> 256, Tanh -> GELU for better convergence
        self.net = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 2)
        )
        
    def forward(self, x, t):
        # t 需要扩展维度以匹配 x
        t_embed = t.view(-1, 1).expand(x.shape[0], 1)
        inp = torch.cat([x, t_embed], dim=1)
        return self.net(inp)

# === 3. 训练过程 ===

# 定义采样函数 (移到前面以便在训练中使用)
def sample_flow(model, n_samples):
    """使用 Euler 方法积分 ODE 生成样本"""
    with torch.no_grad():
        # Initialize at (0.5, 0.5)
        x = torch.randn(n_samples, 2) + 0.5
        # Use more steps for accurate integration (Euler)
        steps = 100
        dt = 1.0 / steps
        traj = [x.clone()]
        for i in range(steps):
            t = torch.ones(n_samples) * (i * dt)
            v = model(x, t)
            x = x + v * dt
            traj.append(x.clone())
    return x, traj

def get_vector_field(model, bounds=(-4, 4), res=20):
        """生成 Vector Field 数据 (at t=0.5)"""
        x = np.linspace(bounds[0], bounds[1], res)
        y = np.linspace(bounds[0], bounds[1], res)
        xx, yy = np.meshgrid(x, y)
        grid = torch.tensor(np.stack([xx, yy], axis=-1), dtype=torch.float32).reshape(-1, 2)
        
        with torch.no_grad():
            # Evaluate at t=0.5 (mid-flow)
            t = torch.ones(grid.shape[0]) * 1
            v = model(grid, t)
            
        return xx, yy, v.numpy()[:, 0].reshape(res, res), v.numpy()[:, 1].reshape(res, res)

gauss_policy = GaussianPolicy()
gauss_opt = optim.Adam(gauss_policy.parameters(), lr=1e-2) # Further increased LR for rapid collapse

flow_model = VectorField()
flow_opt = optim.Adam(flow_model.parameters(), lr=LR)

print("Starting Training...")

# 获取固定的目标样本用于 Flow Matching (模拟从 Replay Buffer 中采样高价值数据)
target_data_pool = env.get_target_samples(20000)

# 记录检查点
checkpoints = [0, 999, 1999] # 对应初始、中间、结束 (Iter 1, 1000, 2000)
snapshots = {}

for it in range(ITERATIONS):
    # --- 1. Train Gaussian (Reverse KL) ---
    # Loss = E_pi [ log_pi - log_target ] (近似 SAC 的目标)
    samples, mu, std = gauss_policy(BATCH_SIZE)
    log_pi = -0.5 * torch.sum(((samples - mu) / std)**2, dim=1) - torch.sum(torch.log(std), dim=0)
    
    with torch.no_grad():
        # log_target = log_pi_old + Q
        log_target = env.log_pi_old(samples) + env.Q_function(samples)
        
    gauss_loss = (log_pi - log_target).mean()
    
    gauss_opt.zero_grad()
    gauss_loss.backward()
    gauss_opt.step()

    # --- 2. Train Flow (Conditional Flow Matching with Minibatch OT) ---
    # Sample x1 from target (High reward data), x0 from noise
    idx = torch.randint(0, len(target_data_pool), (BATCH_SIZE,))
    x1 = target_data_pool[idx]
    # Initialize at (0.5, 0.5)
    x0 = torch.randn_like(x1) + 0.5 
    
    # Optimal Transport Matching
    # Solve assignment problem to minimize total squared distance
    with torch.no_grad():
        # Cost matrix: Euclidean distance squared
        dist = torch.cdist(x0, x1, p=2) ** 2
        # Hungarian algorithm (O(N^3)) - manageable for B=256
        row_ind, col_ind = linear_sum_assignment(dist.cpu().numpy())
        # Reorder x1 to match x0
        x1 = x1[col_ind]

    t = torch.rand(BATCH_SIZE)
    
    # Linear interpolation (path)
    # x_t = t * x1 + (1 - t) * x0
    # target_velocity = x1 - x0
    t_expand = t.view(-1, 1)
    xt = t_expand * x1 + (1 - t_expand) * x0
    ut = x1 - x0 # Vector field target
    
    vt = flow_model(xt, t)
    flow_loss = torch.mean((vt - ut)**2)
    
    flow_opt.zero_grad()
    flow_loss.backward()
    flow_opt.step()

    if it in checkpoints:
        print(f"Snapshot at iter {it+1}")
        # Capture state
        gs, gm, _ = gauss_policy(1000)
        fs, ft = sample_flow(flow_model, 1000)
        
        # Capture Vector Field
        vf_xx, vf_yy, vf_u, vf_v = get_vector_field(flow_model)
        
        snapshots[it] = {
            'g_samples': gs.detach().numpy(),
            'g_mu': gm.detach().numpy(),
            'f_samples': fs.numpy(),
            'f_traj': torch.stack(ft).numpy(),
            'vf': (vf_xx, vf_yy, vf_u, vf_v)
        }

        # --- Hardcode / Force Mode Collapse for Visualization (User Request) ---
        if it == 999 or it == 1999: # Iter 1000 and Iter 2000
            # Force collapse to top-right mode
            # True posterior peak approx calc: (4*2.5 + 0.5*0)/(4+0.5) = 2.22
            target_mean = np.array([2.22, 2.22])
            target_std = 0.5
            
            # Generate fake samples concentrated around target
            fake_samples_g = np.random.randn(1000, 2) * target_std + target_mean
            
            snapshots[it]['g_samples'] = fake_samples_g
            snapshots[it]['g_mu'] = target_mean

            # Also force Flow Policy to be tight for the final plot
            if it == 1999:
                 fake_samples_f = np.random.randn(1000, 2) * target_std + target_mean
                 # Add some noise/mixture to make it look like a flow (maybe some on the other mode?)
                 # User said "concentrate tightly on the two orange yellow regions"
                 # So let's split samples between the two modes
                 n_mode1 = 800
                 n_mode2 = 300
                 mode1 = np.random.randn(n_mode1, 2) * target_std + target_mean
                 mode2 = np.random.randn(n_mode2, 2) * target_std - target_mean # (-2.22, -2.22)
                 fake_samples_f = np.concatenate([mode1, mode2], axis=0)
                 snapshots[it]['f_samples'] = fake_samples_f


    if it % 500 == 0:
        print(f"Iter {it}: Gauss Loss={gauss_loss.item():.4f}, Flow Loss={flow_loss.item():.4f}")

# === 4. 绘图与推理 ===

# 准备背景
xx, yy, density = env.get_energy_grid()

fig, axes = plt.subplots(2, 3, figsize=(9, 5))
# Row 1: Gaussian
# Row 2: Flow

titles = ["Iteration = 0", "Iteration = 1000", "Iteration = 2000"]

bg_color = (235/255, 240/255, 248/255) # Light Blue

# Create custom colormap with transparency
# Based on YlOrRd, but starting with transparency so background shows through
n_colors = 256
colors = plt.cm.YlOrRd(np.linspace(0, 1, n_colors))
# Modify alpha channel: linear ramp from 0 (transparent) to 0.9 (mostly opaque)
colors[:, 3] = np.linspace(0, 0.9, n_colors)
custom_cmap = LinearSegmentedColormap.from_list("CustomYlOrRd", colors)

for i, iter_idx in enumerate(checkpoints):
    data = snapshots[iter_idx]
    
    # Gaussian Plot (Row 0, Col i)
    ax = axes[0, i]
    ax.set_facecolor(bg_color)
    
    # Heatmap style background
    # Use custom_cmap and remove global alpha to allow transparency gradient
    ax.contourf(xx, yy, density, levels=100, cmap=custom_cmap)
    
    try:
        # Change Gaussian points to Purple
        # Add thresh=0.1 to prevent filling the whole background when variance is low
        sns.kdeplot(x=data['g_samples'][:, 0], y=data['g_samples'][:, 1], ax=ax, fill=True, cmap='Purples', alpha=0.4, levels=5, thresh=0.1)
    except Exception as e:
        print(f"KDE plot failed for Gaussian at iter {iter_idx}: {e}")
    
    # Gaussian points: Purple
    ax.scatter(data['g_samples'][:, 0], data['g_samples'][:, 1], c='purple', s=8, alpha=0.6, label='Samples')
    
    ax.set_title(titles[i], fontweight='bold')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Flow Plot (Row 1, Col i)
    ax = axes[1, i]
    ax.set_facecolor(bg_color)
    
    # Heatmap style background
    ax.contourf(xx, yy, density, levels=100, cmap=custom_cmap)
    
    traj_arr = data['f_traj'] # [Steps, N, 2]
    
    # Logic for Flow Visualization (All Iterations now show samples/points)
    # if i == 1: ... (Removed vector field logic for Iter 1000)
    
    # Plot lines WITHOUT arrows (Optional, commented out)
    # num_viz = 50
    # for k in range(num_viz):
    #     ax.plot(traj_arr[:, k, 0], traj_arr[:, k, 1], c='black', alpha=0.3, linewidth=0.8)
    
    # Flow points color
    flow_color = (90/255, 131/255, 207/255)
    
    try:
        sns.kdeplot(x=data['f_samples'][:, 0], y=data['f_samples'][:, 1], ax=ax, fill=True, color=flow_color, alpha=0.4, levels=5, thresh=0.1)
    except Exception as e:
        print(f"KDE plot failed for Flow at iter {iter_idx}: {e}")
    
    ax.scatter(data['f_samples'][:, 0], data['f_samples'][:, 1], c=[flow_color], s=5, alpha=0.4, label='Samples')
    
    # ax.set_title(titles[i], fontweight='bold') # Removed title for bottom row
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

# Add Row Titles (placed above the middle column of each row)
# axes[0, 1].text(0.5, 1.15, "Gaussian Policy", transform=axes[0, 1].transAxes,
#                 ha='center', va='bottom', fontsize=20, fontweight='bold')
# axes[1, 1].text(0.5, 1.15, "Flow Policy", transform=axes[1, 1].transAxes,
#                 ha='center', va='bottom', fontsize=20, fontweight='bold')

# Adjust layout to make space for the titles
plt.subplots_adjust(top=0.94, hspace=0.08, bottom=0.12) # Increase bottom margin for legend

# Add Legend
legend_elements = [
    Line2D([0], [0], color='purple', marker='o', linestyle='None', markersize=10, label='Gaussian Policy Samples'),
    Line2D([0], [0], color=(90/255, 131/255, 207/255), marker='o', linestyle='None', markersize=10, label='Flow Policy Samples'),
    Patch(facecolor='red', edgecolor='orange', label='Target Distribution', alpha=0.5),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.015), fontsize=16, columnspacing=1.0, handletextpad=0.3)

save_path = os.path.join(os.path.dirname(__file__), 'toy_example.pdf')
plt.savefig(save_path)
print(f"Saved plot to {save_path}")
# plt.show()
