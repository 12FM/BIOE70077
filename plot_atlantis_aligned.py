"""
绘制 Atlantis DQN vs DDQN 对比图 - X轴对齐版本
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DQN_COLOR = '#E57836'
DDQN_COLOR = '#4A90D9'

def load_data(log_dir, tag):
    """加载TensorBoard数据"""
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # 尝试精确匹配
        if tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            return steps, values
        
        # 尝试不区分大小写
        for t in ea.Tags().get('scalars', []):
            if t.lower() == tag.lower():
                events = ea.Scalars(t)
                steps = np.array([e.step for e in events])
                values = np.array([e.value for e in events])
                return steps, values
        
        return None, None
    except:
        return None, None

def smooth(values, weight=0.9):
    """指数移动平均平滑"""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        s = last * weight + (1 - weight) * v
        smoothed.append(s)
        last = s
    return np.array(smoothed)

def align_data(steps1, vals1, steps2, vals2):
    """对齐两组数据到相同的X轴范围"""
    # 找到共同的步数范围
    min_step = max(steps1.min(), steps2.min())
    max_step = min(steps1.max(), steps2.max())
    
    # 裁剪数据到共同范围
    mask1 = (steps1 >= min_step) & (steps1 <= max_step)
    mask2 = (steps2 >= min_step) & (steps2 <= max_step)
    
    return steps1[mask1], vals1[mask1], steps2[mask2], vals2[mask2]

print("="*80)
print("绘制 Atlantis DQN vs DDQN 对比图（X轴对齐）")
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
    # 对齐数据
    dqn_steps_aligned, dqn_vals_aligned, ddqn_steps_aligned, ddqn_vals_aligned = \
        align_data(dqn_steps, dqn_vals, ddqn_steps, ddqn_vals)
    
    ax = axes[0]
    dqn_smooth = smooth(dqn_vals_aligned)
    ddqn_smooth = smooth(ddqn_vals_aligned)
    
    ax.plot(dqn_steps_aligned/1e6, dqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN', alpha=0.9)
    ax.plot(ddqn_steps_aligned/1e6, ddqn_smooth, color=DDQN_COLOR, linewidth=2, label='Double DQN', alpha=0.9)
    ax.fill_between(dqn_steps_aligned/1e6, dqn_smooth*0.95, dqn_smooth*1.05, color=DQN_COLOR, alpha=0.2)
    ax.fill_between(ddqn_steps_aligned/1e6, ddqn_smooth*0.95, ddqn_smooth*1.05, color=DDQN_COLOR, alpha=0.2)
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=11)
    ax.set_ylabel('Average Q Value', fontsize=11)
    ax.set_title('Q Value Estimation', fontsize=12, fontweight='bold')
    
    # 添加最终值标注
    dqn_final = dqn_smooth[-1]
    ddqn_final = ddqn_smooth[-1]
    ax.text(0.02, 0.98, f'DQN: {dqn_final:.2f}\nDDQN: {ddqn_final:.2f}', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    print(f"  原始 - DQN: {len(dqn_vals)} 点, DDQN: {len(ddqn_vals)} 点")
    print(f"  对齐后 - DQN: {len(dqn_vals_aligned)} 点, DDQN: {len(ddqn_vals_aligned)} 点")
    print(f"  X轴范围: {dqn_steps_aligned.min()/1e6:.3f} - {dqn_steps_aligned.max()/1e6:.3f} M")

# 图2: Reward
print("\n绘制 Reward...")
dqn_steps, dqn_vals = load_data(dqn_dir, 'Latest 100 avg reward (clipped)')
ddqn_steps, ddqn_vals = load_data(ddqn_dir, 'Latest 100 avg reward (clipped)')

if dqn_steps is not None and ddqn_steps is not None:
    # 对齐数据
    dqn_steps_aligned, dqn_vals_aligned, ddqn_steps_aligned, ddqn_vals_aligned = \
        align_data(dqn_steps, dqn_vals, ddqn_steps, ddqn_vals)
    
    ax = axes[1]
    dqn_smooth = smooth(dqn_vals_aligned)
    ddqn_smooth = smooth(ddqn_vals_aligned)
    
    ax.plot(dqn_steps_aligned/1e6, dqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN', alpha=0.9)
    ax.plot(ddqn_steps_aligned/1e6, ddqn_smooth, color=DDQN_COLOR, linewidth=2, label='Double DQN', alpha=0.9)
    ax.fill_between(dqn_steps_aligned/1e6, 0, dqn_smooth, color=DQN_COLOR, alpha=0.2)
    ax.fill_between(ddqn_steps_aligned/1e6, 0, ddqn_smooth, color=DDQN_COLOR, alpha=0.2)
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=11)
    ax.set_ylabel('Average Reward', fontsize=11)
    ax.set_title('Training Performance', fontsize=12, fontweight='bold')
    
    # 添加最终值标注
    dqn_final = dqn_smooth[-1]
    ddqn_final = ddqn_smooth[-1]
    ax.text(0.02, 0.98, f'DQN: {dqn_final:.2f}\nDDQN: {ddqn_final:.2f}', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    print(f"  对齐后 X轴范围: {dqn_steps_aligned.min()/1e6:.3f} - {dqn_steps_aligned.max()/1e6:.3f} M")

# 图3: Test Score
print("\n绘制 Test Score...")
dqn_steps, dqn_vals = load_data(dqn_dir, 'Test Score')
ddqn_steps, ddqn_vals = load_data(ddqn_dir, 'Test Score')

if dqn_steps is not None and ddqn_steps is not None:
    # 对齐数据
    dqn_steps_aligned, dqn_vals_aligned, ddqn_steps_aligned, ddqn_vals_aligned = \
        align_data(dqn_steps, dqn_vals, ddqn_steps, ddqn_vals)
    
    ax = axes[2]
    dqn_smooth = smooth(dqn_vals_aligned)
    ddqn_smooth = smooth(ddqn_vals_aligned)
    
    ax.plot(dqn_steps_aligned/1e6, dqn_smooth, color=DQN_COLOR, linewidth=2, label='DQN', alpha=0.9)
    ax.plot(ddqn_steps_aligned/1e6, ddqn_smooth, color=DDQN_COLOR, linewidth=2, label='Double DQN', alpha=0.9)
    ax.fill_between(dqn_steps_aligned/1e6, 0, dqn_smooth, color=DQN_COLOR, alpha=0.2)
    ax.fill_between(ddqn_steps_aligned/1e6, 0, ddqn_smooth, color=DDQN_COLOR, alpha=0.2)
    
    ax.set_xlabel('Training Steps (Millions)', fontsize=11)
    ax.set_ylabel('Test Score', fontsize=11)
    ax.set_title('Test Performance', fontsize=12, fontweight='bold')
    
    # 添加最终值标注
    dqn_final = dqn_smooth[-1]
    ddqn_final = ddqn_smooth[-1]
    ax.text(0.02, 0.98, f'DQN: {dqn_final:.0f}\nDDQN: {ddqn_final:.0f}', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    print(f"  对齐后 X轴范围: {dqn_steps_aligned.min()/1e6:.3f} - {dqn_steps_aligned.max()/1e6:.3f} M")

plt.tight_layout()
output = '1_atlantis_core_aligned.png'
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✅ 图已保存: {output}")
print("="*80)
