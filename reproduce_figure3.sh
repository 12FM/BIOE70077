#!/bin/bash
# ============================================================================
# 复现论文图3实验脚本
# Deep Reinforcement Learning with Double Q-learning (arXiv:1509.06461)
# 
# 图3内容：
# - 顶部两行：DQN（橙色）与 Double DQN（蓝色）的价值估计对比
# - 底部一行：实际游戏得分对比
# - 游戏：Alien, Space Invaders, Time Pilot, Zaxxon, Wizard of Wor, Asterix
# ============================================================================

# 激活环境
source /root/miniconda3/bin/activate deep

# 设置 cuDNN 环境变量
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/deep/lib/python3.8/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/deep/lib/python3.8/site-packages/nvidia/cublas/lib

# 进入项目目录
cd /root/Deep

# 定义图3中的6个游戏
GAMES=(
    "AlienNoFrameskip-v4"
    "SpaceInvadersNoFrameskip-v4"
    "TimePilotNoFrameskip-v4"
    "ZaxxonNoFrameskip-v4"
    "WizardOfWorNoFrameskip-v4"
    "AsterixNoFrameskip-v4"
)

# 定义算法
ALGORITHMS=("dqn" "ddqn")

echo "=============================================="
echo "开始复现论文图3实验"
echo "共 ${#GAMES[@]} 个游戏 × ${#ALGORITHMS[@]} 个算法 = $((${#GAMES[@]} * ${#ALGORITHMS[@]})) 个实验"
echo "开始时间: $(date)"
echo "=============================================="

# 循环训练每个游戏的两种算法
for game in "${GAMES[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
        echo ""
        echo "=========================================="
        echo "游戏: $game"
        echo "算法: $algo"
        echo "开始时间: $(date)"
        echo "=========================================="
        
        python main.py --env "$game" --algorithm "$algo" --train --log_interval 100 --save_weight_interval 1000
        
        echo "完成: $game ($algo)"
        echo "结束时间: $(date)"
        echo ""
    done
done

echo "=============================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
echo ""
echo "查看结果:"
echo "  tensorboard --logdir=./log/ --host 0.0.0.0 --port 6006"
echo "=============================================="
