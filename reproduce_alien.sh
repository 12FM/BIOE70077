#!/bin/bash
# 快速复现一个游戏（Alien）的 DQN vs DDQN 对比图

source /root/miniconda3/bin/activate deep
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/deep/lib/python3.8/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/deep/lib/python3.8/site-packages/nvidia/cublas/lib

cd /root/Deep

echo "=============================================="
echo "复现 Alien 游戏的 DQN vs Double DQN 对比"
echo "开始时间: $(date)"
echo "=============================================="

# 训练 DQN
echo ""
echo "[1/2] 训练 DQN..."
python main.py --env AlienNoFrameskip-v4 --algorithm dqn --train --log_interval 100 --save_weight_interval 500

# 训练 Double DQN
echo ""
echo "[2/2] 训练 Double DQN..."
python main.py --env AlienNoFrameskip-v4 --algorithm ddqn --train --log_interval 100 --save_weight_interval 500

echo ""
echo "=============================================="
echo "训练完成!"
echo "结束时间: $(date)"
echo ""
echo "生成对比图:"
echo "  python plot_figure3.py"
echo "=============================================="
