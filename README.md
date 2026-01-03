# 复现论文：Deep Reinforcement Learning with Double Q-learning

> **论文**：[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
>
> **图内容**：DQN（橙色）与 Double DQN（蓝色）在Atari游戏上的对比

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [依赖安装](#2-依赖安装)
3. [Atari ROMs 配置](#3-atari-roms-配置)
4. [GPU 支持配置](#4-gpu-支持配置)
5. [复现图实验](#5-复现图实验)
6. [生成图](#6-生成图)
7. [常见问题](#7-常见问题)

---

## 1. 环境准备

> 🎯 **目标**：创建 Python 3.8 虚拟环境

```bash
# 步骤 1.1：创建环境
conda create -n deep python=3.8 -y

# 步骤 1.2：激活环境
conda activate deep
```

✅ **验证**：命令行前缀显示 `(deep)`

---

## 2. 依赖安装

> 🎯 **目标**：安装训练所需的 Python 包

```bash
# 步骤 2.1：安装 gym
pip install gym==0.15.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 步骤 2.2：安装主要依赖
pip install imageio tensorflow numpy opencv-python matplotlib atari-py -i https://pypi.tuna.tsinghua.edu.cn/simple

# 步骤 2.3：安装 logger
pip install logger -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 3. Atari ROMs 配置

> 🎯 **目标**：下载游戏 ROM 文件

```bash
# 步骤 3.1：安装 AutoROM
pip install autorom[accept-rom-license] -i https://pypi.tuna.tsinghua.edu.cn/simple

# 步骤 3.2：下载 ROMs
AutoROM --accept-license

# 步骤 3.3：导入 ROMs
python -m atari_py.import_roms /root/miniconda3/envs/deep/lib/python3.8/site-packages/AutoROM/roms
```

---

## 4. GPU 支持配置

> 🎯 **目标**：配置 cuDNN 启用 GPU 加速

```bash
# 步骤 4.1：安装 cuDNN
pip install nvidia-cudnn-cu11==8.6.0.163 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 步骤 4.2：设置环境变量（每次运行前执行）
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/deep/lib/python3.8/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/deep/lib/python3.8/site-packages/nvidia/cublas/lib
```

---

## 5. 复现图实验

> 🎯 **目标**：在6款游戏上分别训练 DQN 和 Double DQN，复现论文
>
> **图展示的内容**：
> - 顶部两行：价值估计（Average Q value）对比
> - 底部一行：实际游戏得分对比
> - 橙色 = DQN，蓝色 = Double DQN

### 📊 实验游戏列表（共6个）

| 游戏 | 环境名称 |
|------|----------|
| Alien | `AlienNoFrameskip-v4` |
| Space Invaders | `SpaceInvadersNoFrameskip-v4` |
| Time Pilot | `TimePilotNoFrameskip-v4` |
| Zaxxon | `ZaxxonNoFrameskip-v4` |
| Wizard of Wor | `WizardOfWorNoFrameskip-v4` |
| Asterix | `AsterixNoFrameskip-v4` |

---

### 方法一：一键运行全部实验（推荐）

```bash
cd /root/Deep

# 后台运行全部实验（6游戏 × 2算法 = 12个实验）
nohup ./reproduce_figure3.sh > figure3_training.log 2>&1 &

# 查看训练进度
tail -f figure3_training.log
```

---

### 方法二：逐个运行实验

#### 步骤 5.1：训练 Alien

```bash
cd /root/Deep

# DQN
python main.py --env AlienNoFrameskip-v4 --algorithm dqn --train --log_interval 100 --save_weight_interval 1000

# Double DQN
python main.py --env AlienNoFrameskip-v4 --algorithm ddqn --train --log_interval 100 --save_weight_interval 1000
```

#### 步骤 5.2：训练 Space Invaders

```bash
# DQN
python main.py --env SpaceInvadersNoFrameskip-v4 --algorithm dqn --train --log_interval 100 --save_weight_interval 1000

# Double DQN
python main.py --env SpaceInvadersNoFrameskip-v4 --algorithm ddqn --train --log_interval 100 --save_weight_interval 1000
```

#### 步骤 5.3：训练 Time Pilot

```bash
# DQN
python main.py --env TimePilotNoFrameskip-v4 --algorithm dqn --train --log_interval 100 --save_weight_interval 1000

# Double DQN
python main.py --env TimePilotNoFrameskip-v4 --algorithm ddqn --train --log_interval 100 --save_weight_interval 1000
```

#### 步骤 5.4：训练 Zaxxon

```bash
# DQN
python main.py --env ZaxxonNoFrameskip-v4 --algorithm dqn --train --log_interval 100 --save_weight_interval 1000

# Double DQN
python main.py --env ZaxxonNoFrameskip-v4 --algorithm ddqn --train --log_interval 100 --save_weight_interval 1000
```

#### 步骤 5.5：训练 Wizard of Wor

```bash
# DQN
python main.py --env WizardOfWorNoFrameskip-v4 --algorithm dqn --train --log_interval 100 --save_weight_interval 1000

# Double DQN
python main.py --env WizardOfWorNoFrameskip-v4 --algorithm ddqn --train --log_interval 100 --save_weight_interval 1000
```

#### 步骤 5.6：训练 Asterix

```bash
# DQN
python main.py --env AsterixNoFrameskip-v4 --algorithm dqn --train --log_interval 100 --save_weight_interval 1000

# Double DQN
python main.py --env AsterixNoFrameskip-v4 --algorithm ddqn --train --log_interval 100 --save_weight_interval 1000
```

---

### ⏱️ 预计训练时间

| 项目 | 时间（RTX 3090） |
|------|-----------------|
| 每个实验 | 约 10-20 小时 |
| 全部12个实验 | 约 5-10 天 |

---

## 6. 生成图

> 🎯 **目标**：训练完成后，生成论文的复现图

### 步骤 6.1：查看 TensorBoard（实时监控）

```bash
tensorboard --logdir=./log/ --host 0.0.0.0 --port 6006
```

访问：http://localhost:6006/

### 步骤 6.2：生成图

```bash
python plot_figure3.py
```

输出文件：`./figure3_reproduction.png`

---

## 7. 常见问题

### ❌ ROM is missing

```bash
pip install autorom[accept-rom-license] -i https://pypi.tuna.tsinghua.edu.cn/simple
AutoROM --accept-license
python -m atari_py.import_roms /root/miniconda3/envs/deep/lib/python3.8/site-packages/AutoROM/roms
```

### ❌ DNN library is not found

```bash
pip install nvidia-cudnn-cu11==8.6.0.163 -i https://pypi.tuna.tsinghua.edu.cn/simple
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/deep/lib/python3.8/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/deep/lib/python3.8/site-packages/nvidia/cublas/lib
```

---

## 📚 参考

- 论文：[arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
- 项目文件：
  - `main.py` - 训练入口，支持 `--algorithm dqn/ddqn`
  - `reproduce_figure3.sh` - 一键复现脚本
  - `plot_figure3.py` - 绘制图
