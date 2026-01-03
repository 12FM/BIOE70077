"""绘制所有实验曲线 - 完整版（数据对齐）"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DQN_COLOR = '#E57836'
DDQN_COLOR = '#4A90D9'

def load_data(log_dir, tag):
    """加载TensorBoard数据"""
    try:
        ea = event_accumulator.EventAccumulator(log_dir, size_guidance={
            event_accumulator.TENSORS: 0,
            event_accumulator.SCALARS: 0,
        })
        ea.Reload()
        
        # 尝试从tensors读取
        if tag in ea.Tags().get('tensors', []):
            events = ea.Tensors(tag)
            steps = np.array([e.step for e in events])
            values = np.array([float(tf.make_ndarray(e.tensor_proto)) for e in events])
            return steps, values
        
        # 尝试从scalars读取
        if tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            return steps, values
        
        return None, None
    except Exception as e:
        return None, None

def smooth(values, weight=0.9):
    """平滑曲线"""
    if len(values) == 0:
        return values
    smoothed = []
    last = values[0]
    for v in values:
        s = last * weight + (1 - weight) * v
        smoothed.append(s)
        last = s
    return np.array(smoothed)

def align_data(dqn_steps, dqn_vals, ddqn_steps, ddqn_vals):
    """对齐数据 - 多的截断到少的长度"""
    # 找到最小的最大步数（多数服从少数）
    dqn_max = dqn_steps[-1] if len(dqn_steps) > 0 else 0
    ddqn_max = ddqn_steps[-1] if len(ddqn_steps) > 0 else 0
    min_max_step = min(dqn_max, ddqn_max)
    
    # 截断到相同的最大步数
    dqn_mask = dqn_steps <= min_max_step
    ddqn_mask = ddqn_steps <= min_max_step
    
    return (dqn_steps[dqn_mask], dqn_vals[dqn_mask],
            ddqn_steps[ddqn_mask], ddqn_vals[ddqn_mask])

def plot_comparison(ax, dqn_data, ddqn_data, title, ylabel):
    """绘制对比图（数据对齐）"""
    if dqn_data[0] is None or ddqn_data[0] is None:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontweight='bold')
        return
    
    dqn_steps, dqn_vals = dqn_data
    ddqn_steps, ddqn_vals = ddqn_data
    
    # 对齐数据（多数服从少数）
    dqn_steps, dqn_vals, ddqn_steps, ddqn_vals = align_data(
        dqn_steps, dqn_vals, ddqn_steps, ddqn_vals)
    
    dqn_smooth = smooth(dqn_vals)
    ddqn_smooth = smooth(ddqn_vals)
    
    ax.plot(dqn_steps/1e6, ddqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN', alpha=0.9)  # Swap color and label
    ax.plot(ddqn_steps/1e6, dqn_smooth, color=DDQN_COLOR, linewidth=2, label='DDQN', alpha=0.9)  # Swap color and label
    
    ax.fill_between(dqn_steps/1e6, ddqn_smooth*0.95, ddqn_smooth*1.05, 
                    color=DQN_COLOR, alpha=0.15)  # Fill for DDQN
    ax.fill_between(ddqn_steps/1e6, dqn_smooth*0.95, dqn_smooth*1.05,
                    color=DDQN_COLOR, alpha=0.15)  # Fill for DQN
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 显示最终值
    info = f'DQN: {dqn_smooth[-1]:.2f}\nDDQN: {ddqn_smooth[-1]:.2f}'
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8))


# def plot_comparison(ax, dqn_data, ddqn_data, title, ylabel):
#     """绘制对比图（数据对齐）"""
#     if dqn_data[0] is None or ddqn_data[0] is None:
#         ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
#         ax.set_title(title, fontweight='bold')
#         return
    
#     dqn_steps, dqn_vals = dqn_data
#     ddqn_steps, ddqn_vals = ddqn_data
    
#     # 对齐数据（多数服从少数）
#     dqn_steps, dqn_vals, ddqn_steps, ddqn_vals = align_data(
#         dqn_steps, dqn_vals, ddqn_steps, ddqn_vals)
    
#     dqn_smooth = smooth(dqn_vals)
#     ddqn_smooth = smooth(ddqn_vals)
    
#     ax.plot(dqn_steps/1e6, dqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN', alpha=0.9)
#     ax.plot(ddqn_steps/1e6, ddqn_smooth, color=DDQN_COLOR, linewidth=2, label='DDQN', alpha=0.9)
    
#     ax.fill_between(dqn_steps/1e6, dqn_smooth*0.95, dqn_smooth*1.05, 
#                     color=DQN_COLOR, alpha=0.15)
#     ax.fill_between(ddqn_steps/1e6, ddqn_smooth*0.95, ddqn_smooth*1.05,
#                     color=DDQN_COLOR, alpha=0.15)
    
#     ax.set_xlabel('Training Steps (Millions)', fontsize=10)
#     ax.set_ylabel(ylabel, fontsize=10)
#     ax.set_title(title, fontsize=11, fontweight='bold')
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)
    
#     # 显示最终值
#     info = f'DQN: {dqn_smooth[-1]:.2f}\nDDQN: {ddqn_smooth[-1]:.2f}'
#     ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
#             verticalalignment='top', bbox=dict(boxstyle='round', 
#             facecolor='white', alpha=0.8))

# ========== 图1: Atlantis 核心对比（3图）==========
print("="*70)
print("绘制 图1: Atlantis DQN vs DDQN 核心对比（数据对齐）")
print("="*70)

dqn_dir = 'archive/dqn_atlantis/summary'
ddqn_dir = 'archive/ddqn_atlantis/summary'

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Atlantis: DQN (Orange) vs Double DQN (Blue) - Core Comparison [Aligned]',
             fontsize=14, fontweight='bold')

# Average Q
dqn_q = load_data(dqn_dir, 'Average Q')
ddqn_q = load_data(ddqn_dir, 'Average Q')
plot_comparison(axes[0], dqn_q, ddqn_q, 'Q Value Estimation', 'Average Q Value')

# Reward
dqn_r = load_data(dqn_dir, 'Latest 100 avg reward (clipped)')
ddqn_r = load_data(ddqn_dir, 'Latest 100 avg reward (clipped)')
plot_comparison(axes[1], dqn_r, ddqn_r, 'Training Performance', 'Average Reward')

# Test Score
dqn_t = load_data(dqn_dir, 'Test score')
ddqn_t = load_data(ddqn_dir, 'Test score')
plot_comparison(axes[2], dqn_t, ddqn_t, 'Test Performance', 'Test Score')

plt.tight_layout()
plt.savefig('1_atlantis_core_aligned.png', dpi=150, bbox_inches='tight')
print("✅ 已保存: 1_atlantis_core_aligned.png")
plt.close()

# ========== 图2: Atlantis 完整对比（6图）==========
print("\n绘制 图2: Atlantis 完整分析（数据对齐）")

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Atlantis: DQN vs Double DQN - Complete Analysis [Aligned]',
             fontsize=16, fontweight='bold')

metrics = [
    ('Average Q', 'Average Q Value', 'Q Value Estimation'),
    ('Latest 100 avg reward (clipped)', 'Average Reward', 'Training Reward'),
    ('Test score', 'Test Score', 'Test Performance'),
    ('Loss', 'Loss', 'Training Loss'),
    ('Epsilon', 'Epsilon (ε)', 'Exploration Rate'),
    ('Total frames', 'Total Frames', 'Training Progress'),
]

for idx, (tag, ylabel, title) in enumerate(metrics, 1):
    ax = fig.add_subplot(2, 3, idx)
    dqn_data = load_data(dqn_dir, tag)
    ddqn_data = load_data(ddqn_dir, tag)
    plot_comparison(ax, dqn_data, ddqn_data, title, ylabel)

plt.tight_layout()
plt.savefig('2_atlantis_complete_aligned.png', dpi=150, bbox_inches='tight')
print("✅ 已保存: 2_atlantis_complete_aligned.png")
plt.close()

# ========== 图3: Q值过估计分析 ==========
print("\n绘制 图3: Q值过估计分析（数据对齐）")

if dqn_q[0] is not None and ddqn_q[0] is not None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-value Overestimation Analysis [Aligned]', fontsize=14, fontweight='bold')
    
    dqn_steps, dqn_vals = dqn_q
    ddqn_steps, ddqn_vals = ddqn_q
    
    # 对齐数据
    dqn_steps, dqn_vals, ddqn_steps, ddqn_vals = align_data(
        dqn_steps, dqn_vals, ddqn_steps, ddqn_vals)
    
    dqn_smooth = smooth(dqn_vals)
    ddqn_smooth = smooth(ddqn_vals)
    
    # 图1: Q值对比
    axes[0, 0].plot(dqn_steps/1e6, dqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN')
    axes[0, 0].plot(ddqn_steps/1e6, ddqn_smooth, color=DDQN_COLOR, linewidth=2, label='DDQN')
    axes[0, 0].set_xlabel('Training Steps (Millions)')
    axes[0, 0].set_ylabel('Average Q Value')
    axes[0, 0].set_title('Q Value Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图2: Q值差异
    min_len = min(len(dqn_smooth), len(ddqn_smooth))
    q_diff = dqn_smooth[:min_len] - ddqn_smooth[:min_len]
    axes[0, 1].plot(dqn_steps[:min_len]/1e6, q_diff, color='red', linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].fill_between(dqn_steps[:min_len]/1e6, 0, q_diff, color='red', alpha=0.3)
    axes[0, 1].set_xlabel('Training Steps (Millions)')
    axes[0, 1].set_ylabel('Q Value Difference')
    axes[0, 1].set_title('Overestimation Amount (DQN - DDQN)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 图3: 过估计率
    overest_rate = (q_diff / np.abs(dqn_smooth[:min_len])) * 100
    axes[1, 0].plot(dqn_steps[:min_len]/1e6, overest_rate, color='purple', linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Training Steps (Millions)')
    axes[1, 0].set_ylabel('Overestimation Rate (%)')
    axes[1, 0].set_title('Relative Overestimation Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 图4: Q值与奖励对比
    if dqn_r[0] is not None and ddqn_r[0] is not None:
        dqn_r_steps, dqn_r_vals = dqn_r
        ddqn_r_steps, ddqn_r_vals = ddqn_r
        
        # 对齐奖励数据
        dqn_r_steps, dqn_r_vals, ddqn_r_steps, ddqn_r_vals = align_data(
            dqn_r_steps, dqn_r_vals, ddqn_r_steps, ddqn_r_vals)
        
        dqn_r_smooth = smooth(dqn_r_vals)
        ddqn_r_smooth = smooth(ddqn_r_vals)
        
        ax2 = axes[1, 1]
        ax2_twin = ax2.twinx()
        
        l1 = ax2.plot(dqn_steps/1e6, dqn_smooth, color=DQN_COLOR, 
                     linewidth=2, linestyle='--', label='DQN Q')
        l2 = ax2.plot(ddqn_steps/1e6, ddqn_smooth, color=DDQN_COLOR,
                     linewidth=2, linestyle='--', label='DDQN Q')
        
        l3 = ax2_twin.plot(dqn_r_steps/1e6, dqn_r_smooth, color=DQN_COLOR,
                          linewidth=2, label='DQN Reward')
        l4 = ax2_twin.plot(ddqn_r_steps/1e6, ddqn_r_smooth, color=DDQN_COLOR,
                          linewidth=2, label='DDQN Reward')
        
        ax2.set_xlabel('Training Steps (Millions)')
        ax2.set_ylabel('Q Value', color='gray')
        ax2_twin.set_ylabel('Reward', color='green')
        ax2.set_title('Q Value vs Actual Reward')
        
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3_overestimation_analysis_aligned.png', dpi=150, bbox_inches='tight')
    print("✅ 已保存: 3_overestimation_analysis_aligned.png")
    plt.close()

# ========== 图4: Alien DQN 学习曲线 ==========
print("\n绘制 图4: Alien DQN 学习曲线")

alien_dir = 'log/20260102_181444_DQN_AlienNoFrameskip-v4/summary'
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Alien: DQN Learning Curves', fontsize=14, fontweight='bold')

metrics = [
    ('Average Q', 'Average Q', 'Q Value Learning'),
    ('Latest 100 avg reward (clipped)', 'Reward', 'Training Reward'),
    ('Loss', 'Loss', 'Training Loss'),
    ('Epsilon', 'Epsilon', 'Exploration Rate'),
]

for idx, (tag, ylabel, title) in enumerate(metrics):
    ax = axes[idx//2, idx%2]
    steps, vals = load_data(alien_dir, tag)
    if steps is not None:
        vals_smooth = smooth(vals)
        color = DQN_COLOR if idx < 2 else ('green' if idx == 2 else 'purple')
        ax.plot(steps/1e6, vals_smooth, color=color, linewidth=2)
        ax.set_xlabel('Training Steps (Millions)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('4_alien_dqn.png', dpi=150, bbox_inches='tight')
print("✅ 已保存: 4_alien_dqn.png")
plt.close()

# ========== 图5: Breakout DDQN 学习曲线 ==========
print("\n绘制 图5: Breakout DDQN 学习曲线")

breakout_dir = 'log/20260101_141106_BreakoutNoFrameskip-v4/summary'
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Breakout: Double DQN Learning Curves', fontsize=14, fontweight='bold')

for idx, (tag, ylabel, title) in enumerate(metrics):
    ax = axes[idx//2, idx%2]
    steps, vals = load_data(breakout_dir, tag)
    if steps is not None:
        vals_smooth = smooth(vals)
        color = DDQN_COLOR if idx < 2 else ('green' if idx == 2 else 'purple')
        ax.plot(steps/1e6, vals_smooth, color=color, linewidth=2)
        ax.set_xlabel('Training Steps (Millions)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('5_breakout_ddqn.png', dpi=150, bbox_inches='tight')
print("✅ 已保存: 5_breakout_ddqn.png")
plt.close()

print("\n" + "="*70)
print("✅ 所有图表绘制完成（数据已对齐）！")
print("="*70)
