"""
用现有训练数据绘制 DQN vs DDQN 对比图
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色定义
DQN_COLOR = '#E57836'      # 橙色
DDQN_COLOR = '#4A90D9'     # 蓝色

def load_tensorboard_data(log_dir, tag):
    """从TensorBoard日志加载数据"""
    try:
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={
                event_accumulator.TENSORS: 0,
                event_accumulator.SCALARS: 0,
            }
        )
        ea.Reload()
        
        # 从 tensors 读取（TF2 格式）
        if tag in ea.Tags().get('tensors', []):
            events = ea.Tensors(tag)
            steps = np.array([e.step for e in events])
            values = np.array([float(tf.make_ndarray(e.tensor_proto)) for e in events])
            return steps, values
        
        # 从 scalars 读取（TF1 格式）
        if tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            return steps, values
        
        return None, None
    except Exception as e:
        print(f"  读取 {tag} 出错: {e}")
        return None, None

def smooth_data(values, weight=0.9):
    """指数移动平均平滑"""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def find_all_logs(log_base="./log"):
    """查找所有训练日志"""
    results = []
    for d in glob.glob(f"{log_base}/*/summary/"):
        parent = os.path.dirname(os.path.dirname(d))
        name = os.path.basename(parent)
        
        if "_DQN_" in name:
            algo = "DQN"
            game = name.split("_DQN_")[1]
        else:
            algo = "DDQN"
            parts = name.split("_")
            if len(parts) >= 2:
                game = "_".join(parts[2:]) if len(parts) > 2 else parts[1]
            else:
                game = name
        
        event_files = glob.glob(f"{d}/events.out.*")
        if event_files and os.path.getsize(event_files[0]) > 100000:  # >100KB 才算有效
            results.append({
                'path': d,
                'name': name,
                'game': game,
                'algorithm': algo,
                'size': os.path.getsize(event_files[0])
            })
    
    return results

def plot_existing_data(log_base="./log"):
    """绘制现有数据"""
    logs = find_all_logs(log_base)
    
    print("=== 找到的有效训练日志 ===")
    for log in sorted(logs, key=lambda x: x['game']):
        print(f"  {log['algorithm']:5} | {log['game']:30} | {log['size']/1024/1024:.1f} MB")
    print()
    
    games = {}
    for log in sorted(logs, key=lambda x: x['name']):
        game = log['game']
        if game not in games:
            games[game] = {'DQN': None, 'DDQN': None}
        games[game][log['algorithm']] = log['path']
    
    valid_games = {k: v for k, v in games.items() if v['DQN'] or v['DDQN']}
    
    if not valid_games:
        print("没有找到有效的训练数据！")
        return
    
    print("=== 将要绘制的游戏 ===")
    for game, paths in valid_games.items():
        dqn_status = "✓" if paths['DQN'] else "✗"
        ddqn_status = "✓" if paths['DDQN'] else "✗"
        print(f"  {game}: DQN={dqn_status}, DDQN={ddqn_status}")
    print()
    
    n_games = len(valid_games)
    fig, axes = plt.subplots(2, n_games, figsize=(6*n_games, 10))
    fig.suptitle('Figure 3: DQN (orange) vs Double DQN (blue) - Value Estimates and Scores', fontsize=14, fontweight='bold')
    
    if n_games == 1:
        axes = axes.reshape(-1, 1)
    
    game_names = list(valid_games.keys())
    
    for col, game in enumerate(game_names):
        paths = valid_games[game]
        display_name = game.replace("NoFrameskip-v4", "")
        
        # 第1行：Average Q value
        ax1 = axes[0, col]
        ax1.set_title(display_name, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value Estimate\n(Average Q)', fontsize=10)
        ax1.set_xlabel('Training Steps (episodes)', fontsize=9)
        
        # DQN
        if paths['DQN']:
            steps, values = load_tensorboard_data(paths['DQN'], 'Average Q')
            if steps is not None and len(steps) > 10:
                print(f"  DQN {display_name} Average Q: {len(steps)} 点, 范围 [{values.min():.2f}, {values.max():.2f}]")
                smoothed = smooth_data(values)
                ax1.plot(steps, smoothed, color=DQN_COLOR, linewidth=2, label='DQN')
                ax1.fill_between(steps, smoothed - np.std(values)*0.3, smoothed + np.std(values)*0.3, 
                                color=DQN_COLOR, alpha=0.2)
        
        # DDQN
        if paths['DDQN']:
            steps, values = load_tensorboard_data(paths['DDQN'], 'Average Q')
            if steps is not None and len(steps) > 10:
                print(f"  DDQN {display_name} Average Q: {len(steps)} 点, 范围 [{values.min():.2f}, {values.max():.2f}]")
                smoothed = smooth_data(values)
                ax1.plot(steps, smoothed, color=DDQN_COLOR, linewidth=2, label='Double DQN')
                ax1.fill_between(steps, smoothed - np.std(values)*0.3, smoothed + np.std(values)*0.3,
                                color=DDQN_COLOR, alpha=0.2)
        
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 第2行：Score (Reward)
        ax2 = axes[1, col]
        ax2.set_ylabel('Score\n(Reward)', fontsize=10)
        ax2.set_xlabel('Training Steps (episodes)', fontsize=9)
        
        # DQN
        if paths['DQN']:
            steps, values = load_tensorboard_data(paths['DQN'], 'Latest 100 avg reward (clipped)')
            if steps is not None and len(steps) > 10:
                print(f"  DQN {display_name} Reward: {len(steps)} 点, 范围 [{values.min():.2f}, {values.max():.2f}]")
                smoothed = smooth_data(values)
                ax2.plot(steps, smoothed, color=DQN_COLOR, linewidth=2, label='DQN')
                ax2.fill_between(steps, 0, smoothed, color=DQN_COLOR, alpha=0.2)
        
        # DDQN
        if paths['DDQN']:
            steps, values = load_tensorboard_data(paths['DDQN'], 'Latest 100 avg reward (clipped)')
            if steps is not None and len(steps) > 10:
                print(f"  DDQN {display_name} Reward: {len(steps)} 点, 范围 [{values.min():.2f}, {values.max():.2f}]")
                smoothed = smooth_data(values)
                ax2.plot(steps, smoothed, color=DDQN_COLOR, linewidth=2, label='Double DQN')
                ax2.fill_between(steps, 0, smoothed, color=DDQN_COLOR, alpha=0.2)
        
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = './existing_data_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图已保存到: {output_path}")

if __name__ == "__main__":
    plot_existing_data("./log")
