"""
绘制论文Figure 3风格的曲线图
展示Q值估计 vs 真实值（True Value）
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

DQN_COLOR = '#E57836'      # 橙色
DDQN_COLOR = '#4A90D9'     # 蓝色
DQN_TRUE_COLOR = '#D32F2F'   # 深红色
DDQN_TRUE_COLOR = '#1976D2'  # 深蓝色

def load_data(log_dir, tag):
    """加载TensorBoard数据"""
    try:
        ea = event_accumulator.EventAccumulator(log_dir, size_guidance={
            event_accumulator.TENSORS: 0,
            event_accumulator.SCALARS: 0,
        })
        ea.Reload()
        
        if tag in ea.Tags().get('tensors', []):
            events = ea.Tensors(tag)
            steps = np.array([e.step for e in events])
            values = np.array([float(tf.make_ndarray(e.tensor_proto)) for e in events])
            return steps, values
        
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
    dqn_max = dqn_steps[-1] if len(dqn_steps) > 0 else 0
    ddqn_max = ddqn_steps[-1] if len(ddqn_steps) > 0 else 0
    min_max_step = min(dqn_max, ddqn_max)
    
    dqn_mask = dqn_steps <= min_max_step
    ddqn_mask = ddqn_steps <= min_max_step
    
    return (dqn_steps[dqn_mask], dqn_vals[dqn_mask],
            ddqn_steps[ddqn_mask], ddqn_vals[ddqn_mask])

def plot_paper_style(ax, dqn_q_data, ddqn_q_data, dqn_reward_data, ddqn_reward_data, game_name):
    """
    绘制论文风格的曲线
    - Q值估计曲线（estimate）
    - 真实值横线（true value，用reward的后期平均值表示）
    """
    
    if dqn_q_data[0] is None or ddqn_q_data[0] is None:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(game_name, fontweight='bold')
        return
    
    # 对齐Q值数据
    dqn_q_steps, dqn_q_vals = dqn_q_data
    ddqn_q_steps, ddqn_q_vals = ddqn_q_data
    dqn_q_steps, dqn_q_vals, ddqn_q_steps, ddqn_q_vals = align_data(
        dqn_q_steps, dqn_q_vals, ddqn_q_steps, ddqn_q_vals)
    
    # 平滑Q值
    dqn_q_smooth = smooth(dqn_q_vals)
    ddqn_q_smooth = smooth(ddqn_q_vals)
    
    # 计算真实值（用reward的后20%数据的平均值）
    dqn_true_value = None
    ddqn_true_value = None
    
    if dqn_reward_data[0] is not None:
        _, dqn_r_vals = dqn_reward_data
        cutoff = int(len(dqn_r_vals) * 0.8)
        dqn_true_value = np.mean(dqn_r_vals[cutoff:])
    
    if ddqn_reward_data[0] is not None:
        _, ddqn_r_vals = ddqn_reward_data
        cutoff = int(len(ddqn_r_vals) * 0.8)
        ddqn_true_value = np.mean(ddqn_r_vals[cutoff:])
    
    # 转换为百万步
    dqn_steps_m = dqn_q_steps / 1e6
    ddqn_steps_m = ddqn_q_steps / 1e6
    
    # 绘制DQN估计（橙色，上方）
    ax.plot(dqn_steps_m, dqn_q_smooth, color=DQN_COLOR, linewidth=2, 
            label='DQN estimate', alpha=0.9, zorder=3)
    ax.fill_between(dqn_steps_m, dqn_q_smooth * 0.9, dqn_q_smooth * 1.1,
                    color=DQN_COLOR, alpha=0.2, zorder=1)
    
    # 绘制DDQN估计（蓝色，下方）
    ax.plot(ddqn_steps_m, ddqn_q_smooth, color=DDQN_COLOR, linewidth=2,
            label='Double DQN estimate', alpha=0.9, zorder=3)
    ax.fill_between(ddqn_steps_m, ddqn_q_smooth * 0.9, ddqn_q_smooth * 1.1,
                    color=DDQN_COLOR, alpha=0.2, zorder=1)
    
    # 绘制DQN真实值横线（红色）
    if dqn_true_value is not None:
        ax.axhline(y=dqn_true_value, color=DQN_TRUE_COLOR, linestyle='-',
                  linewidth=1.5, label='DQN true value', alpha=0.8, zorder=2)
    
    # 绘制DDQN真实值横线（深蓝色）
    if ddqn_true_value is not None:
        ax.axhline(y=ddqn_true_value, color=DDQN_TRUE_COLOR, linestyle='-',
                  linewidth=1.5, label='Double DQN true value', alpha=0.8, zorder=2)
    
    ax.set_xlabel('Training steps (in millions)', fontsize=9)
    ax.set_ylabel('Value estimates', fontsize=9)
    ax.set_title(game_name, fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

print("="*70)
print("绘制论文Figure 3风格的曲线")
print("="*70)

# 读取Atlantis数据
dqn_dir = 'archive/dqn_atlantis/summary'
ddqn_dir = 'archive/ddqn_atlantis/summary'

print("\n加载数据...")
dqn_q = load_data(dqn_dir, 'Average Q')
ddqn_q = load_data(ddqn_dir, 'Average Q')
dqn_reward = load_data(dqn_dir, 'Latest 100 avg reward (clipped)')
ddqn_reward = load_data(ddqn_dir, 'Latest 100 avg reward (clipped)')

print(f"  DQN Q值: {len(dqn_q[0]) if dqn_q[0] is not None else 0} 点")
print(f"  DDQN Q值: {len(ddqn_q[0]) if ddqn_q[0] is not None else 0} 点")
print(f"  DQN Reward: {len(dqn_reward[0]) if dqn_reward[0] is not None else 0} 点")
print(f"  DDQN Reward: {len(ddqn_reward[0]) if ddqn_reward[0] is not None else 0} 点")

# 创建单图
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig.suptitle('Atlantis: Q Value Estimates vs True Values', 
             fontsize=14, fontweight='bold')

plot_paper_style(ax, dqn_q, ddqn_q, dqn_reward, ddqn_reward, 'Atlantis')

plt.tight_layout()
plt.savefig('atlantis_paper_style.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n✅ 已保存: atlantis_paper_style.png")
plt.close()

# 如果有多个游戏，可以创建1x4的布局（模拟论文Figure 3）
# 目前只有Atlantis数据，所以只画1个
print("\n提示: 如果要复现完整的论文Figure 3（4个游戏），需要以下数据:")
print("  - Space Invaders: DQN + DDQN")
print("  - Time Pilot: DQN + DDQN")
print("  - Zaxxon: DQN + DDQN")
print("  目前只有 Atlantis 的完整数据")

print("\n" + "="*70)
