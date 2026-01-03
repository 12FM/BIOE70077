"""
绘制论文图3：DQN vs Double DQN 对比图

论文图3布局：
- 第1行（4列）：Alien, Space Invaders, Time Pilot, Zaxxon 的 Q值估计
- 第2行（2列）：Wizard of Wor, Asterix 的 Q值估计  
- 第3行（2列）：Wizard of Wor, Asterix 的实际得分

颜色：
- DQN = 橙色 (orange)
- Double DQN = 蓝色 (blue)
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 游戏列表（与论文图3一致）
GAMES_ROW1 = ['Alien', 'Space Invaders', 'Time Pilot', 'Zaxxon']
GAMES_ROW2_3 = ['Wizard of Wor', 'Asterix']

GAME_ENVS = {
    'Alien': 'AlienNoFrameskip-v4',
    'Space Invaders': 'SpaceInvadersNoFrameskip-v4',
    'Time Pilot': 'TimePilotNoFrameskip-v4',
    'Zaxxon': 'ZaxxonNoFrameskip-v4',
    'Wizard of Wor': 'WizardOfWorNoFrameskip-v4',
    'Asterix': 'AsterixNoFrameskip-v4',
}

# 颜色定义（与论文一致）
DQN_COLOR = '#E57836'      # 橙色
DDQN_COLOR = '#4A90D9'     # 蓝色
DQN_FILL = '#F5C4A1'       # 浅橙色
DDQN_FILL = '#A8CCE8'      # 浅蓝色


def load_tensorboard_data(log_dir, tag):
    """从TensorBoard日志加载数据"""
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        if tag not in ea.Tags()['scalars']:
            return None, None
        
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        return steps, values
    except:
        return None, None


def find_log_dir(base_path, game_env, algorithm):
    """查找对应的日志目录"""
    if algorithm == "dqn":
        pattern = f"{base_path}/*_DQN_{game_env}"
    else:
        # DDQN日志不包含"DQN_"前缀
        dirs = glob.glob(f"{base_path}/*_{game_env}")
        dirs = [d for d in dirs if "_DQN_" not in d]
        if dirs:
            return sorted(dirs)[-1] + "/summary/"
        return None
    
    dirs = glob.glob(pattern)
    if dirs:
        return sorted(dirs)[-1] + "/summary/"
    return None


def smooth_data(values, weight=0.9):
    """指数移动平均平滑"""
    if len(values) == 0:
        return values
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_game_q_value(ax, log_base_path, game_name, show_ylabel=False):
    """绘制单个游戏的Q值对比"""
    game_env = GAME_ENVS[game_name]
    dqn_dir = find_log_dir(log_base_path, game_env, "dqn")
    ddqn_dir = find_log_dir(log_base_path, game_env, "ddqn")
    
    ax.set_title(game_name, fontsize=11, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Value Estimate', fontsize=10)
    ax.set_xlabel('Training steps (in millions)', fontsize=9)
    
    # 绘制 DQN（橙色）
    if dqn_dir and os.path.exists(dqn_dir):
        steps, values = load_tensorboard_data(dqn_dir, 'Average Q')
        if steps is not None and len(steps) > 0:
            # 转换为百万步
            steps_m = steps / 1e6
            smoothed = smooth_data(values)
            ax.plot(steps_m, smoothed, color=DQN_COLOR, linewidth=1.5, label='DQN')
            ax.fill_between(steps_m, smoothed * 0.8, smoothed * 1.2, 
                          color=DQN_FILL, alpha=0.5)
    
    # 绘制 Double DQN（蓝色）
    if ddqn_dir and os.path.exists(ddqn_dir):
        steps, values = load_tensorboard_data(ddqn_dir, 'Average Q')
        if steps is not None and len(steps) > 0:
            steps_m = steps / 1e6
            smoothed = smooth_data(values)
            ax.plot(steps_m, smoothed, color=DDQN_COLOR, linewidth=1.5, label='Double DQN')
            ax.fill_between(steps_m, smoothed * 0.8, smoothed * 1.2,
                          color=DDQN_FILL, alpha=0.5)
    
    ax.grid(True, alpha=0.3)


def plot_game_score(ax, log_base_path, game_name, show_ylabel=False):
    """绘制单个游戏的得分对比"""
    game_env = GAME_ENVS[game_name]
    dqn_dir = find_log_dir(log_base_path, game_env, "dqn")
    ddqn_dir = find_log_dir(log_base_path, game_env, "ddqn")
    
    ax.set_title(game_name, fontsize=11, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Score', fontsize=10)
    ax.set_xlabel('Training steps (in millions)', fontsize=9)
    
    # 绘制 DQN（橙色）
    if dqn_dir and os.path.exists(dqn_dir):
        steps, values = load_tensorboard_data(dqn_dir, 'Latest 100 avg reward (clipped)')
        if steps is not None and len(steps) > 0:
            steps_m = steps / 1e6
            smoothed = smooth_data(values)
            ax.plot(steps_m, smoothed, color=DQN_COLOR, linewidth=1.5, label='DQN')
            ax.fill_between(steps_m, 0, smoothed, color=DQN_FILL, alpha=0.5)
    
    # 绘制 Double DQN（蓝色）
    if ddqn_dir and os.path.exists(ddqn_dir):
        steps, values = load_tensorboard_data(ddqn_dir, 'Latest 100 avg reward (clipped)')
        if steps is not None and len(steps) > 0:
            steps_m = steps / 1e6
            smoothed = smooth_data(values)
            ax.plot(steps_m, smoothed, color=DDQN_COLOR, linewidth=1.5, label='Double DQN')
            ax.fill_between(steps_m, 0, smoothed, color=DDQN_FILL, alpha=0.5)
    
    ax.grid(True, alpha=0.3)


def plot_figure3(log_base_path="./log"):
    """绘制论文图3"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 第1行：4个游戏的Q值估计（Alien, Space Invaders, Time Pilot, Zaxxon）
    for i, game in enumerate(GAMES_ROW1):
        ax = fig.add_subplot(3, 4, i + 1)
        plot_game_q_value(ax, log_base_path, game, show_ylabel=(i == 0))
    
    # 第2行：Wizard of Wor 和 Asterix 的Q值估计
    ax_wiz_q = fig.add_subplot(3, 4, 5)
    plot_game_q_value(ax_wiz_q, log_base_path, 'Wizard of Wor', show_ylabel=True)
    
    ax_ast_q = fig.add_subplot(3, 4, 6)
    plot_game_q_value(ax_ast_q, log_base_path, 'Asterix', show_ylabel=False)
    
    # 第3行：Wizard of Wor 和 Asterix 的得分
    ax_wiz_s = fig.add_subplot(3, 4, 9)
    plot_game_score(ax_wiz_s, log_base_path, 'Wizard of Wor', show_ylabel=True)
    
    ax_ast_s = fig.add_subplot(3, 4, 10)
    plot_game_score(ax_ast_s, log_base_path, 'Asterix', show_ylabel=False)
    
    # 添加图例
    handles = [
        plt.Line2D([0], [0], color=DQN_COLOR, linewidth=2, label='DQN'),
        plt.Line2D([0], [0], color=DDQN_COLOR, linewidth=2, label='Double DQN')
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=12, frameon=True)
    
    # 添加标题
    fig.suptitle('Figure 3: DQN (orange) vs Double DQN (blue) - Value Estimates and Scores', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    output_path = './figure3_reproduction.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"图3已保存到: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    plot_figure3("./log")
