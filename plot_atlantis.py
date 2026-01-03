"""绘制 Atlantis DQN vs DDQN 对比图 - 横坐标对齐版本"""
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
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={
                event_accumulator.TENSORS: 0,
                event_accumulator.SCALARS: 0,
            }
        )
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
        print(f"  读取 {tag} 失败: {e}")
        return None, None

def smooth(values, weight=0.9):
    """平滑曲线"""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        s = last * weight + (1 - weight) * v
        smoothed.append(s)
        last = s
    return np.array(smoothed)

def truncate_to_match(dqn_steps, ddqn_steps, ddqn_values):
    """截断DDQN数据以匹配DQN的最大步数"""
    max_dqn_step = dqn_steps.max()
    mask = ddqn_steps <= max_dqn_step
    return ddqn_steps[mask], ddqn_values[mask]

print("="*80)
print("绘制 Atlantis DQN vs DDQN 对比图 (横坐标对齐)")
print("="*80)

dqn_dir = 'archive/dqn_atlantis/summary'
ddqn_dir = 'archive/ddqn_atlantis/summary'

# 创建3图对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Atlantis: DQN (Orange) vs Double DQN (Blue) - Core Comparison', fontsize=14, fontweight='bold')

# 图1: Average Q
print("\n绘制 Average Q...")
dqn_steps, dqn_vals = load_data(dqn_dir, 'Average Q')
ddqn_steps, ddqn_vals = load_data(ddqn_dir, 'Average Q')

if dqn_steps is not None and ddqn_steps is not None:
    # 截断DDQN数据
    ddqn_steps, ddqn_vals = truncate_to_match(dqn_steps, ddqn_steps, ddqn_vals)
    
    ax = axes[0]
    dqn_smooth = smooth(dqn_vals)
    ddqn_smooth = smooth(ddqn_vals)
    
    # 添加文本框显示最终值
    textstr = f'DQN: {dqn_smooth[-1]:.2f}\nDDQN: {ddqn_smooth[-1]:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    ax.plot(dqn_steps/1e6, dqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN', alpha=0.8)
    ax.plot(ddqn_steps/1e6, ddqn_smooth, color=DDQN_COLOR, linewidth=2, label='Double DQN', alpha=0.8)
    ax.fill_between(dqn_steps/1e6, dqn_smooth*0.95, dqn_smooth*1.05, color=DQN_COLOR, alpha=0.2)
    ax.fill_between(ddqn_steps/1e6, ddqn_smooth*0.95, ddqn_smooth*1.05, color=DDQN_COLOR, alpha=0.2)
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=11)
    ax.set_ylabel('Average Q Value', fontsize=11)
    ax.set_title('Q Value Estimation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    max_step = dqn_steps.max() / 1e6
    print(f"  DQN: {len(dqn_vals)} 点, 最大步数={max_step:.3f}M, 最终值={dqn_smooth[-1]:.2f}")
    print(f"  DDQN: {len(ddqn_vals)} 点 (截断后), 最大步数={max_step:.3f}M, 最终值={ddqn_smooth[-1]:.2f}")

# 图2: Reward
print("\n绘制 Reward...")
dqn_steps, dqn_vals = load_data(dqn_dir, 'Latest 100 avg reward (clipped)')
ddqn_steps, ddqn_vals = load_data(ddqn_dir, 'Latest 100 avg reward (clipped)')

if dqn_steps is not None and ddqn_steps is not None:
    # 截断DDQN数据
    ddqn_steps, ddqn_vals = truncate_to_match(dqn_steps, ddqn_steps, ddqn_vals)
    
    ax = axes[1]
    dqn_smooth = smooth(dqn_vals)
    ddqn_smooth = smooth(ddqn_vals)
    
    # 添加文本框显示最终值
    textstr = f'DQN: {dqn_smooth[-1]:.2f}\nDDQN: {ddqn_smooth[-1]:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    ax.plot(dqn_steps/1e6, dqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN', alpha=0.8)
    ax.plot(ddqn_steps/1e6, ddqn_smooth, color=DDQN_COLOR, linewidth=2, label='Double DQN', alpha=0.8)
    ax.fill_between(dqn_steps/1e6, 0, dqn_smooth, color=DQN_COLOR, alpha=0.2)
    ax.fill_between(ddqn_steps/1e6, 0, ddqn_smooth, color=DDQN_COLOR, alpha=0.2)
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=11)
    ax.set_ylabel('Average Reward', fontsize=11)
    ax.set_title('Training Performance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    print(f"  DQN: 最终值={dqn_smooth[-1]:.2f}")
    print(f"  DDQN: 最终值={ddqn_smooth[-1]:.2f}")

# 图3: Test Score
print("\n绘制 Test Score...")
dqn_steps, dqn_vals = load_data(dqn_dir, 'Test score')
ddqn_steps, ddqn_vals = load_data(ddqn_dir, 'Test score')

if dqn_steps is not None and ddqn_steps is not None:
    # 截断DDQN数据
    ddqn_steps, ddqn_vals = truncate_to_match(dqn_steps, ddqn_steps, ddqn_vals)
    
    ax = axes[2]
    dqn_smooth = smooth(dqn_vals)
    ddqn_smooth = smooth(ddqn_vals)
    
    # 添加文本框显示最终值
    textstr = f'DQN: {dqn_smooth[-1]:.0f}\nDDQN: {ddqn_smooth[-1]:.0f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    ax.plot(dqn_steps/1e6, dqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN', alpha=0.8)
    ax.plot(ddqn_steps/1e6, ddqn_smooth, color=DDQN_COLOR, linewidth=2, label='Double DQN', alpha=0.8)
    ax.fill_between(dqn_steps/1e6, 0, dqn_smooth, color=DQN_COLOR, alpha=0.2)
    ax.fill_between(ddqn_steps/1e6, 0, ddqn_smooth, color=DDQN_COLOR, alpha=0.2)
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=11)
    ax.set_ylabel('Test Score', fontsize=11)
    ax.set_title('Test Performance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    print(f"  DQN: 最终值={dqn_smooth[-1]:.0f}")
    print(f"  DDQN: 最终值={ddqn_smooth[-1]:.0f}")

plt.tight_layout()
output = '1_atlantis_core.png'
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✅ 图已保存: {output}")
print(f"   DQN和DDQN的横坐标已对齐，结尾在相同位置")
plt.close()

print("="*80)
