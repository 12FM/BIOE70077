#!/usr/bin/env python3
"""绘制所有实验曲线"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DQN_COLOR = '#E57836'
DDQN_COLOR = '#4A90D9'

def read_tfevents(log_dir):
    """读取TensorBoard数据"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        data = {}
        for tag in ea.Tags().get('scalars', []):
            try:
                events = ea.Scalars(tag)
                data[tag] = {
                    'steps': np.array([e.step for e in events]),
                    'values': np.array([e.value for e in events])
                }
            except:
                pass
        return data
    except Exception as e:
        print(f"读取失败: {e}")
        return {}

def smooth(values, weight=0.9):
    """平滑曲线"""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        s = last * weight + (1 - weight) * v
        smoothed.append(s)
        last = s
    return np.array(smoothed)

print("="*70)
print("开始绘制实验曲线")
print("="*70)

# 读取Atlantis数据
print("\n1. 读取 Atlantis 数据...")
dqn_data = read_tfevents('archive/dqn_atlantis/summary')
ddqn_data = read_tfevents('archive/ddqn_atlantis/summary')

print(f"  DQN: {len(dqn_data)} 个指标")
print(f"  DDQN: {len(ddqn_data)} 个指标")

if dqn_data and ddqn_data:
    print("\n2. 绘制核心对比图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Atlantis: DQN vs Double DQN', fontsize=14, fontweight='bold')
    
    # Average Q
    if 'Average Q' in dqn_data and 'Average Q' in ddqn_data:
        ax = axes[0]
        dqn_steps = dqn_data['Average Q']['steps'] / 1e6
        dqn_vals = smooth(dqn_data['Average Q']['values'])
        ddqn_steps = ddqn_data['Average Q']['steps'] / 1e6
        ddqn_vals = smooth(ddqn_data['Average Q']['values'])
        
        ax.plot(dqn_steps, dqn_vals, color=DQN_COLOR, linewidth=2, label='DQN')
        ax.plot(ddqn_steps, ddqn_vals, color=DDQN_COLOR, linewidth=2, label='DDQN')
        ax.set_xlabel('Training Steps (M)')
        ax.set_ylabel('Average Q')
        ax.set_title('Q Value Estimation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Reward
    reward_tag = 'Latest 100 avg reward (clipped)'
    if reward_tag in dqn_data and reward_tag in ddqn_data:
        ax = axes[1]
        dqn_steps = dqn_data[reward_tag]['steps'] / 1e6
        dqn_vals = smooth(dqn_data[reward_tag]['values'])
        ddqn_steps = ddqn_data[reward_tag]['steps'] / 1e6
        ddqn_vals = smooth(ddqn_data[reward_tag]['values'])
        
        ax.plot(dqn_steps, dqn_vals, color=DQN_COLOR, linewidth=2, label='DQN')
        ax.plot(ddqn_steps, ddqn_vals, color=DDQN_COLOR, linewidth=2, label='DDQN')
        ax.set_xlabel('Training Steps (M)')
        ax.set_ylabel('Reward')
        ax.set_title('Training Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Test Score
    if 'Test Score' in dqn_data and 'Test Score' in ddqn_data:
        ax = axes[2]
        dqn_steps = dqn_data['Test Score']['steps'] / 1e6
        dqn_vals = smooth(dqn_data['Test Score']['values'])
        ddqn_steps = ddqn_data['Test Score']['steps'] / 1e6
        ddqn_vals = smooth(ddqn_data['Test Score']['values'])
        
        ax.plot(dqn_steps, dqn_vals, color=DQN_COLOR, linewidth=2, label='DQN')
        ax.plot(ddqn_steps, ddqn_vals, color=DDQN_COLOR, linewidth=2, label='DDQN')
        ax.set_xlabel('Training Steps (M)')
        ax.set_ylabel('Test Score')
        ax.set_title('Test Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output = 'atlantis_comparison.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"  ✅ 已保存: {output}")
    plt.close()

# 读取Alien DQN数据
print("\n3. 读取 Alien DQN 数据...")
alien1 = read_tfevents('log/20260102_181444_DQN_AlienNoFrameskip-v4/summary')
alien2 = read_tfevents('log/20260101_160626_DQN_AlienNoFrameskip-v4/summary')

if alien1 or alien2:
    print(f"  Run1: {len(alien1)} 个指标")
    print(f"  Run2: {len(alien2)} 个指标")
    
    print("\n4. 绘制 Alien DQN 学习曲线...")
    
    best_data = alien1 if len(alien1) > len(alien2) else alien2
    
    if best_data:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Alien: DQN Learning Curves', fontsize=14, fontweight='bold')
        
        # Average Q
        if 'Average Q' in best_data:
            ax = axes[0, 0]
            steps = best_data['Average Q']['steps'] / 1e6
            vals = smooth(best_data['Average Q']['values'])
            ax.plot(steps, vals, color=DQN_COLOR, linewidth=2)
            ax.set_xlabel('Steps (M)')
            ax.set_ylabel('Average Q')
            ax.set_title('Q Value Learning')
            ax.grid(True, alpha=0.3)
        
        # Reward
        if reward_tag in best_data:
            ax = axes[0, 1]
            steps = best_data[reward_tag]['steps'] / 1e6
            vals = smooth(best_data[reward_tag]['values'])
            ax.plot(steps, vals, color=DQN_COLOR, linewidth=2)
            ax.set_xlabel('Steps (M)')
            ax.set_ylabel('Reward')
            ax.set_title('Training Reward')
            ax.grid(True, alpha=0.3)
        
        # Loss
        if 'Loss' in best_data:
            ax = axes[1, 0]
            steps = best_data['Loss']['steps'] / 1e6
            vals = smooth(best_data['Loss']['values'])
            ax.plot(steps, vals, color='green', linewidth=2)
            ax.set_xlabel('Steps (M)')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True, alpha=0.3)
        
        # Epsilon
        if 'Epsilon' in best_data:
            ax = axes[1, 1]
            steps = best_data['Epsilon']['steps'] / 1e6
            vals = best_data['Epsilon']['values']
            ax.plot(steps, vals, color='purple', linewidth=2)
            ax.set_xlabel('Steps (M)')
            ax.set_ylabel('Epsilon')
            ax.set_title('Exploration Rate')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output = 'alien_dqn_learning.png'
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"  ✅ 已保存: {output}")
        plt.close()

# 读取Breakout DDQN数据
print("\n5. 读取 Breakout DDQN 数据...")
breakout = read_tfevents('log/20260101_141106_BreakoutNoFrameskip-v4/summary')

if breakout:
    print(f"  找到 {len(breakout)} 个指标")
    
    print("\n6. 绘制 Breakout DDQN 学习曲线...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Breakout: Double DQN Learning Curves', fontsize=14, fontweight='bold')
    
    # Average Q
    if 'Average Q' in breakout:
        ax = axes[0, 0]
        steps = breakout['Average Q']['steps'] / 1e6
        vals = smooth(breakout['Average Q']['values'])
        ax.plot(steps, vals, color=DDQN_COLOR, linewidth=2)
        ax.set_xlabel('Steps (M)')
        ax.set_ylabel('Average Q')
        ax.set_title('Q Value Learning')
        ax.grid(True, alpha=0.3)
    
    # Reward
    if reward_tag in breakout:
        ax = axes[0, 1]
        steps = breakout[reward_tag]['steps'] / 1e6
        vals = smooth(breakout[reward_tag]['values'])
        ax.plot(steps, vals, color=DDQN_COLOR, linewidth=2)
        ax.set_xlabel('Steps (M)')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward')
        ax.grid(True, alpha=0.3)
    
    # Loss
    if 'Loss' in breakout:
        ax = axes[1, 0]
        steps = breakout['Loss']['steps'] / 1e6
        vals = smooth(breakout['Loss']['values'])
        ax.plot(steps, vals, color='green', linewidth=2)
        ax.set_xlabel('Steps (M)')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
    
    # Epsilon
    if 'Epsilon' in breakout:
        ax = axes[1, 1]
        steps = breakout['Epsilon']['steps'] / 1e6
        vals = breakout['Epsilon']['values']
        ax.plot(steps, vals, color='purple', linewidth=2)
        ax.set_xlabel('Steps (M)')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output = 'breakout_ddqn_learning.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"  ✅ 已保存: {output}")
    plt.close()

print("\n" + "="*70)
print("绘图完成！生成的图表:")
print("  1. atlantis_comparison.png - Atlantis DQN vs DDQN 对比")
print("  2. alien_dqn_learning.png - Alien DQN 学习曲线")
print("  3. breakout_ddqn_learning.png - Breakout DDQN 学习曲线")
print("="*70)
