"""绘制所有实验曲线 - 完整版"""
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

def plot_comparison(ax, dqn_data, ddqn_data, title, ylabel):
    """绘制对比图"""
    if dqn_data[0] is None or ddqn_data[0] is None:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontweight='bold')
        return
    
    dqn_steps, dqn_vals = dqn_data
    ddqn_steps, ddqn_vals = ddqn_data
    
    dqn_smooth = smooth(dqn_vals)
    ddqn_smooth = smooth(ddqn_vals)
    
    ax.plot(dqn_steps/1e6, dqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN', alpha=0.9)
    ax.plot(ddqn_steps/1e6, ddqn_smooth, color=DDQN_COLOR, linewidth=2, label='DDQN', alpha=0.9)
    
    ax.fill_between(dqn_steps/1e6, dqn_smooth*0.95, dqn_smooth*1.05, 
                    color=DQN_COLOR, alpha=0.15)
    ax.fill_between(ddqn_steps/1e6, ddqn_smooth*0.95, ddqn_smooth*1.05,
                    color=DDQN_COLOR, alpha=0.15)
    
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

# ========== 图1: Atlantis 核心对比（3图）==========
print("="*70)
print("绘制 图1: Atlantis DQN vs DDQN 核心对比")
print("="*70)

dqn_dir = 'archive/dqn_atlantis/summary'
ddqn_dir = 'archive/ddqn_atlantis/summary'

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Atlantis: DQN (Orange) vs Double DQN (Blue) - Core Comparison',
             fontsize=14, fontweight='bold')

# Average Q
dqn_q = load_data(dqn_dir, 'Average Q')
ddqn_q = load_data(ddqn_dir, 'Average Q')
plot_comparison(axes[0], dqn_q, ddqn_q, 'Q Value Estimation', 'Average Q Value')

# Reward
dqn_r = load_data(dqn_dir, 'Latest 100 avg reward (clipped)')
ddqn_r = load_data(ddqn_dir, 'Latest 100 avg reward (clipped)')
plot_comparison(axes[1], dqn_r, ddqn_r, 'Training Performance', 'Average Reward')

# Test Score (注意：标签是小写 'Test score')
dqn_t = load_data(dqn_dir, 'Test score')
ddqn_t = load_data(ddqn_dir, 'Test score')
plot_comparison(axes[2], dqn_t, ddqn_t, 'Test Performance', 'Test Score')

plt.tight_layout()
plt.savefig('1_atlantis_core.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ 已保存: 1_atlantis_core.png\n")
plt.close()

# ========== 图2: Atlantis 完整分析（6图）==========
print("="*70)
print("绘制 图2: Atlantis DQN vs DDQN 完整分析")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Atlantis: DQN vs Double DQN - Complete Analysis',
             fontsize=16, fontweight='bold')

# 第一行
plot_comparison(axes[0,0], dqn_q, ddqn_q, 'Q Value', 'Average Q')
plot_comparison(axes[0,1], dqn_r, ddqn_r, 'Reward', 'Avg Reward')
plot_comparison(axes[0,2], dqn_t, ddqn_t, 'Test Score', 'Score')

# 第二行
dqn_loss = load_data(dqn_dir, 'Loss')
ddqn_loss = load_data(ddqn_dir, 'Loss')
plot_comparison(axes[1,0], dqn_loss, ddqn_loss, 'Loss', 'Loss')

dqn_eps = load_data(dqn_dir, 'Epsilon')
ddqn_eps = load_data(ddqn_dir, 'Epsilon')
plot_comparison(axes[1,1], dqn_eps, ddqn_eps, 'Exploration', 'Epsilon')

dqn_frames = load_data(dqn_dir, 'Total Frames')
ddqn_frames = load_data(ddqn_dir, 'Total Frames')
plot_comparison(axes[1,2], dqn_frames, ddqn_frames, 'Training Progress', 'Frames')

plt.tight_layout()
plt.savefig('2_atlantis_complete.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ 已保存: 2_atlantis_complete.png\n")
plt.close()

# ========== 图3: Q值过估计分析 ==========
print("="*70)
print("绘制 图3: Q值过估计分析")
print("="*70)

if dqn_q[0] is not None and ddqn_q[0] is not None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-value Overestimation Analysis', fontsize=14, fontweight='bold')
    
    dqn_steps, dqn_vals = dqn_q
    ddqn_steps, ddqn_vals = ddqn_q
    dqn_smooth = smooth(dqn_vals)
    ddqn_smooth = smooth(ddqn_vals)
    
    # 图1: Q值对比
    axes[0,0].plot(dqn_steps/1e6, dqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN')
    axes[0,0].plot(ddqn_steps/1e6, ddqn_smooth, color=DDQN_COLOR, linewidth=2, label='DDQN')
    axes[0,0].set_xlabel('Steps (M)')
    axes[0,0].set_ylabel('Average Q')
    axes[0,0].set_title('Q Value Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 图2: Q值差异（过估计量）
    min_len = min(len(dqn_smooth), len(ddqn_smooth))
    q_diff = dqn_smooth[:min_len] - ddqn_smooth[:min_len]
    axes[0,1].plot(dqn_steps[:min_len]/1e6, q_diff, color='red', linewidth=2)
    axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0,1].fill_between(dqn_steps[:min_len]/1e6, 0, q_diff, 
                          color='red', alpha=0.3)
    axes[0,1].set_xlabel('Steps (M)')
    axes[0,1].set_ylabel('Q Difference')
    axes[0,1].set_title('Overestimation (DQN - DDQN)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 图3: 过估计率
    overest_rate = (q_diff / np.abs(dqn_smooth[:min_len])) * 100
    axes[1,0].plot(dqn_steps[:min_len]/1e6, overest_rate, color='purple', linewidth=2)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].set_xlabel('Steps (M)')
    axes[1,0].set_ylabel('Overestimation Rate (%)')
    axes[1,0].set_title('Relative Overestimation')
    axes[1,0].grid(True, alpha=0.3)
    
    # 图4: 统计信息
    axes[1,1].axis('off')
    stats_text = f"""
    Q值统计分析
    
    DQN:
      最小: {dqn_smooth.min():.2f}
      最大: {dqn_smooth.max():.2f}
      平均: {dqn_smooth.mean():.2f}
      最终: {dqn_smooth[-1]:.2f}
    
    DDQN:
      最小: {ddqn_smooth.min():.2f}
      最大: {ddqn_smooth.max():.2f}
      平均: {ddqn_smooth.mean():.2f}
      最终: {ddqn_smooth[-1]:.2f}
    
    过估计:
      平均差异: {q_diff.mean():.2f}
      最大差异: {q_diff.max():.2f}
      平均比率: {overest_rate.mean():.1f}%
    """
    axes[1,1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                  verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('3_overestimation_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ 已保存: 3_overestimation_analysis.png\n")
    plt.close()

# ========== 图4: Alien DQN 学习曲线 ==========
print("="*70)
print("绘制 图4: Alien DQN 学习曲线")
print("="*70)

alien_dir = 'log/20260102_181444_DQN_AlienNoFrameskip-v4/summary'

if os.path.exists(alien_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Alien: DQN Learning Curves', fontsize=14, fontweight='bold')
    
    # Q值
    steps, vals = load_data(alien_dir, 'Average Q')
    if steps is not None:
        axes[0,0].plot(steps/1e6, smooth(vals), color=DQN_COLOR, linewidth=2)
        axes[0,0].set_xlabel('Steps (M)')
        axes[0,0].set_ylabel('Average Q')
        axes[0,0].set_title('Q Value Learning')
        axes[0,0].grid(True, alpha=0.3)
    
    # 奖励
    steps, vals = load_data(alien_dir, 'Latest 100 avg reward (clipped)')
    if steps is not None:
        axes[0,1].plot(steps/1e6, smooth(vals), color=DQN_COLOR, linewidth=2)
        axes[0,1].set_xlabel('Steps (M)')
        axes[0,1].set_ylabel('Reward')
        axes[0,1].set_title('Training Reward')
        axes[0,1].grid(True, alpha=0.3)
    
    # 损失
    steps, vals = load_data(alien_dir, 'Loss')
    if steps is not None:
        axes[1,0].plot(steps/1e6, smooth(vals), color='green', linewidth=2)
        axes[1,0].set_xlabel('Steps (M)')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].set_title('Training Loss')
        axes[1,0].grid(True, alpha=0.3)
    
    # Epsilon
    steps, vals = load_data(alien_dir, 'Epsilon')
    if steps is not None:
        axes[1,1].plot(steps/1e6, vals, color='purple', linewidth=2)
        axes[1,1].set_xlabel('Steps (M)')
        axes[1,1].set_ylabel('Epsilon')
        axes[1,1].set_title('Exploration Rate')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('4_alien_dqn.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ 已保存: 4_alien_dqn.png\n")
    plt.close()

# ========== 图5: Breakout DDQN 学习曲线 ==========
print("="*70)
print("绘制 图5: Breakout DDQN 学习曲线")
print("="*70)

breakout_dir = 'log/20260101_141106_BreakoutNoFrameskip-v4/summary'

if os.path.exists(breakout_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Breakout: Double DQN Learning Curves', fontsize=14, fontweight='bold')
    
    # Q值
    steps, vals = load_data(breakout_dir, 'Average Q')
    if steps is not None:
        axes[0,0].plot(steps/1e6, smooth(vals), color=DDQN_COLOR, linewidth=2)
        axes[0,0].set_xlabel('Steps (M)')
        axes[0,0].set_ylabel('Average Q')
        axes[0,0].set_title('Q Value Learning')
        axes[0,0].grid(True, alpha=0.3)
    
    # 奖励
    steps, vals = load_data(breakout_dir, 'Latest 100 avg reward (clipped)')
    if steps is not None:
        axes[0,1].plot(steps/1e6, smooth(vals), color=DDQN_COLOR, linewidth=2)
        axes[0,1].set_xlabel('Steps (M)')
        axes[0,1].set_ylabel('Reward')
        axes[0,1].set_title('Training Reward')
        axes[0,1].grid(True, alpha=0.3)
    
    # 损失
    steps, vals = load_data(breakout_dir, 'Loss')
    if steps is not None:
        axes[1,0].plot(steps/1e6, smooth(vals), color='green', linewidth=2)
        axes[1,0].set_xlabel('Steps (M)')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].set_title('Training Loss')
        axes[1,0].grid(True, alpha=0.3)
    
    # Epsilon
    steps, vals = load_data(breakout_dir, 'Epsilon')
    if steps is not None:
        axes[1,1].plot(steps/1e6, vals, color='purple', linewidth=2)
        axes[1,1].set_xlabel('Steps (M)')
        axes[1,1].set_ylabel('Epsilon')
        axes[1,1].set_title('Exploration Rate')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('5_breakout_ddqn.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ 已保存: 5_breakout_ddqn.png\n")
    plt.close()

print("="*70)
print("所有图表绘制完成！")
print("="*70)
print("\n生成的图表:")
print("  1_atlantis_core.png - Atlantis核心对比（3图）")
print("  2_atlantis_complete.png - Atlantis完整分析（6图）")
print("  3_overestimation_analysis.png - Q值过估计分析（4图）")
print("  4_alien_dqn.png - Alien DQN学习曲线（4图）")
print("  5_breakout_ddqn.png - Breakout DDQN学习曲线（4图）")
print("\n共5张图，21个子图")
print("="*70)
