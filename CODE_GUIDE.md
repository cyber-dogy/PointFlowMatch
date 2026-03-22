# PointFlowMatch 代码完全指南

> 本指南专为具身智能领域初学者编写，详细解释代码架构、数据流和修改方法

---

## 📋 目录

1. [项目概述](#1-项目概述)
2. [代码架构总览](#2-代码架构总览)
3. [详细文件索引](#3-详细文件索引)
4. [核心概念解析](#4-核心概念解析)
5. [模型输入输出详解](#5-模型输入输出详解)
6. [训练流程详解](#6-训练流程详解)
7. [数据结构详解](#7-数据结构详解)
8. [如何修改代码](#8-如何修改代码)
9. [常见问题](#9-常见问题)

---

## 1. 项目概述

### 1.1 这是什么？

**PointFlowMatch (PFP)** 是一个基于**条件流匹配(Conditional Flow Matching)** 的机器人操作学习框架。它直接从点云观测中学习机器人操作策略。

### 1.2 核心特点

| 特点 | 说明 |
|------|------|
| 输入 | 点云 (Point Cloud) + 机器人当前状态 |
| 输出 | 未来动作序列 (端到端位姿 + 夹爪状态) |
| 算法 | 条件流匹配 (Flow Matching) - 比扩散模型更快 |
| 仿真环境 | RLBench |
| 状态表示 | 6D旋转 + 3D位置 + 1D夹爪 |

### 1.3 目录结构

```
PointFlowMatch/
├── pfp/                    # 核心代码库
├── scripts/                # 可执行脚本
├── conf/                   # 配置文件 (Hydra)
├── demos/                  # 演示数据存放目录
├── ckpt/                   # 模型检查点
├── urdfs/                  # 机器人URDF模型
├── bash/                   # Shell脚本
└── toy_circle/             # 简单示例环境
```

---

## 2. 代码架构总览

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         训练流程 (train.py)                       │
├─────────────────────────────────────────────────────────────────┤
│  数据加载                                                        │
│  ├─ RobotDatasetPcd: 从zarr文件加载点云数据                        │
│  └─ DataLoader: PyTorch数据加载器                                │
├─────────────────────────────────────────────────────────────────┤
│  模型 (FMPolicy)                                                │
│  ├─ obs_encoder: 观测编码器 (PointNet/PointMLP/ResNet)           │
│  │   └─ 输入: 点云 (B, T, P, 3/6) + 机器人状态 (B, T, 10)         │
│  │   └─ 输出: 观测特征 (B, T, obs_features_dim)                   │
│  │                                                              │
│  └─ diffusion_net: 流匹配网络 (ConditionalUnet1D)                │
│      └─ 输入: 带噪声动作 z_t (B, T_pred, 10) + 时间步 t + 观测条件   │
│      └─ 输出: 预测速度 (B, T_pred, 10)                            │
├─────────────────────────────────────────────────────────────────┤
│  损失计算                                                        │
│  └─ MSE(pred_velocity, target_velocity)                          │
│      ├─ xyz位置损失                                              │
│      ├─ rot6d旋转损失                                            │
│      └─ gripper夹爪损失                                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         推理流程 (evaluate.py)                    │
├─────────────────────────────────────────────────────────────────┤
│  环境 (RLBenchEnv)                                              │
│  ├─ 获取观测: 点云 + 机器人状态                                   │
│  └─ 执行动作: 通过逆运动学控制机器人                               │
├─────────────────────────────────────────────────────────────────┤
│  策略推理                                                        │
│  └─ infer_y(): 从噪声逐步去噪得到动作序列                          │
│      ├─ 初始化: z_0 ~ 高斯噪声                                    │
│      ├─ 迭代K步: z_{i+1} = z_i + v_pred * dt                     │
│      └─ 输出: 最终动作序列                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 详细文件索引

### 3.1 核心模块 (`pfp/`)

#### 3.1.1 策略网络 (`pfp/policy/`)

| 文件 | 功能 | 关键类 | 修改场景 |
|------|------|--------|----------|
| `base_policy.py` | 策略基类 | `BasePolicy` | 修改观测缓冲逻辑 |
| `fm_policy.py` | **流匹配策略(主)** | `FMPolicy` | **修改训练/推理逻辑** |
| `ddim_policy.py` | DDIM扩散策略 | `DDIMPolicy` | 对比实验 |
| `fm_so3_policy.py` | SO(3)流匹配 | `FMSO3Policy` | 使用四元数旋转表示 |
| `fm_5p_policy.py` | 5点表示法 | `FM5pPolicy` | 使用5点表示末端执行器 |
| `fm_se3_policy.py` | SE(3)流匹配 | `FMSE3Policy` | 显式SE(3)约束 |

#### 3.1.2 骨干网络 (`pfp/backbones/`)

| 文件 | 功能 | 输入 | 输出 | 修改场景 |
|------|------|------|------|----------|
| `pointnet.py` | **PointNet编码器** | (B*T, P, 3/6) | (B, T*(embed_dim+10)) | **修改点云编码** |
| `pointmlp.py` | PointMLP编码器 | (B*T, P, 3/6) | (B, T*(embed_dim+10)) | 更换骨干网络 |
| `resnet_dp.py` | ResNet图像编码 | (B*T, 5, H, W, 3) | (B, T*embed_dim) | 图像输入实验 |
| `mlp_3dp.py` | 简单MLP | - | - | 基线对比 |

#### 3.1.3 数据集 (`pfp/data/`)

| 文件 | 功能 | 数据源 | 修改场景 |
|------|------|--------|----------|
| `replay_buffer.py` | Zarr数据存储 | `demos/sim/` | 修改数据存储格式 |
| `dataset_pcd.py` | **点云数据集** | zarr文件 | **修改数据预处理** |
| `dataset_images.py` | 图像数据集 | zarr文件 | 图像输入实验 |

#### 3.1.4 环境 (`pfp/envs/`)

| 文件 | 功能 | 关键类 | 修改场景 |
|------|------|--------|----------|
| `base_env.py` | 环境基类 | `BaseEnv` | 添加新环境类型 |
| `rlbench_env.py` | **RLBench环境** | `RLBenchEnv` | **修改观测/动作** |
| `rlbench_runner.py` | 评测运行器 | `RLBenchRunner` | 修改评测逻辑 |

#### 3.1.5 工具函数 (`pfp/common/`)

| 文件 | 功能 | 关键函数 | 修改场景 |
|------|------|----------|----------|
| `se3_utils.py` | SE(3)变换工具 | `pfp_to_pose_th`, `rot6d_to_quat_np` | 修改旋转表示 |
| `fm_utils.py` | 流匹配工具 | `get_timesteps` | 修改调度器 |
| `o3d_utils.py` | Open3D工具 | `make_pcd`, `merge_pcds` | 修改点云处理 |
| `visualization.py` | Rerun可视化 | `RerunViewer` | 修改可视化 |

#### 3.1.6 根文件 (`pfp/`)

| 文件 | 功能 | 关键内容 |
|------|------|----------|
| `__init__.py` | 包初始化 | `DATA_DIRS`, `REPO_DIRS`, `DEVICE` |

### 3.2 脚本 (`scripts/`)

| 文件 | 功能 | 使用方法 | 修改场景 |
|------|------|----------|----------|
| `train.py` | **训练主脚本** | `python scripts/train.py task_name=xxx` | **修改训练流程** |
| `evaluate.py` | **评测主脚本** | `python scripts/evaluate.py policy.ckpt_name=xxx` | 修改评测流程 |
| `collect_demos.py` | 收集演示数据 | `bash bash/collect_data.sh` | 修改数据收集 |
| `vis_dataset.py` | 可视化数据集 | - | 调试数据 |
| `convert_data.py` | 数据格式转换 | - | 迁移数据 |

### 3.3 配置文件 (`conf/`)

#### 主配置

| 文件 | 功能 | 关键配置 |
|------|------|----------|
| `train.yaml` | **训练配置** | epochs, batch_size, lr, model, backbone |
| `eval.yaml` | **评测配置** | ckpt_name, num_episodes, vis |
| `collect_demos_train.yaml` | 数据收集(训练集) | num_episodes, task_name |
| `collect_demos_valid.yaml` | 数据收集(验证集) | num_episodes, task_name |

#### 模型配置 (`conf/model/`)

| 文件 | 策略类 | 用途 |
|------|--------|------|
| `flow.yaml` | `FMPolicy` | **标准流匹配(推荐)** |
| `flow_5p.yaml` | `FM5pPolicy` | 5点表示法 |
| `flow_so3.yaml` | `FMSO3Policy` | SO(3)旋转空间 |
| `flow_se3.yaml` | `FMSE3Policy` | SE(3)约束 |
| `ddim.yaml` | `DDIMPolicy` | DDIM基线 |

#### 骨干配置 (`conf/backbone/`)

| 文件 | 用途 |
|------|------|
| `pointnet.yaml` | PointNet编码器 |
| `pointmlp.yaml` | PointMLP编码器 |
| `resnet_dp.yaml` | ResNet图像编码 |
| `mlp_3dp.yaml` | 简单MLP |

#### 实验配置 (`conf/experiment/`)

| 文件 | 用途 |
|------|------|
| `pointflowmatch.yaml` | **主实验(推荐)** |
| `pointflowmatch_images.yaml` | 图像输入实验 |
| `pointflowmatch_so3.yaml` | SO(3)实验 |
| `diffusion_policy.yaml` | Diffusion Policy基线 |
| `dp3.yaml` | 3D Diffusion Policy基线 |

---

## 4. 核心概念解析

### 4.1 机器人状态表示

```python
# robot_state: 10维向量
[px, py, pz, r00, r10, r20, r01, r11, r21, gripper]
#  │   位置(3)  │        旋转6D(6)        │   夹爪(1)  │

# 旋转6D -> 3x3旋转矩阵 (使用Gram-Schmidt正交化)
# r00,r10,r20: 旋转矩阵第一列
# r01,r11,r21: 旋转矩阵第二列
# 第三列 = 第一列 × 第二列
```

### 4.2 点云表示

```python
# 点云形状: (Batch, Time, Points, Channels)
# - Batch: 批量大小
# - Time: 观测时间步 (n_obs_steps=2)
# - Points: 点数 (n_points=4096)
# - Channels: 3(仅xyz) 或 6(xyz+rgb)

pcd.shape = (128, 2, 4096, 3)  # 训练时
pcd.shape = (1, 2, 4096, 3)    # 推理时
```

### 4.3 流匹配(Flow Matching)原理

```
训练:
1. 从数据分布采样: z_1 ~ p_data(y)
2. 从高斯分布采样: z_0 ~ N(0, I)
3. 线性插值: z_t = t * z_1 + (1-t) * z_0,  t~U(0,1)
4. 目标速度: v = z_1 - z_0
5. 网络预测: v_pred = model(z_t, t, condition)
6. 损失: MSE(v_pred, v)

推理:
1. 初始化: z = z_0 ~ N(0, I)
2. for i in range(K):
     v = model(z, t[i], condition)
     z = z + v * dt[i]
3. 输出: z 即为预测动作
```

### 4.4 条件信息(Conditioning)

```python
# 条件 = 观测编码 + 机器人状态
nx = obs_encoder(pcd, robot_state_obs)
# nx.shape = (Batch, n_obs_steps * (obs_features_dim + y_dim))
# 默认: (128, 2 * (256 + 10)) = (128, 532)
```

---

## 5. 模型输入输出详解

### 5.1 FMPolicy (主策略类)

```python
# ==================== 初始化参数 ====================
FMPolicy(
    x_dim=266,              # 观测特征维度 = obs_features_dim + y_dim
    y_dim=10,               # 动作维度 = xyz(3) + rot6d(6) + gripper(1)
    n_obs_steps=2,          # 观测时间步数
    n_pred_steps=32,        # 预测时间步数
    num_k_infer=10,         # 推理时的积分步数
    time_conditioning=True, # 是否使用时间条件
    obs_encoder=PointNetBackbone(...),  # 观测编码器
    diffusion_net=ConditionalUnet1D(...),  # 流匹配网络
    ...
)

# ==================== 训练输入 ====================
# batch = (pcd, robot_state_obs, robot_state_pred)
pcd:              torch.Tensor  # (B, n_obs_steps, n_points, 3/6)
robot_state_obs:  torch.Tensor  # (B, n_obs_steps, 10)
robot_state_pred: torch.Tensor  # (B, n_pred_steps, 10)

# ==================== 训练输出 ====================
loss: torch.Tensor  # 标量损失值

# ==================== 推理输入 ====================
pcd:              torch.Tensor  # (B, n_obs_steps, n_points, 3/6)
robot_state_obs:  torch.Tensor  # (B, n_obs_steps, 10)

# ==================== 推理输出 ====================
robot_state_pred: torch.Tensor  # (K, B, n_pred_steps, 10)
# K: 积分步数+1 (包含初始噪声和最终预测)
```

### 5.2 PointNetBackbone (观测编码器)

```python
# ==================== 输入 ====================
pcd:              torch.Tensor  # (B, T, P, C)  C=3或6
robot_state_obs:  torch.Tensor  # (B, T, 10)

# ==================== 处理流程 ====================
# 1. Flatten: (B, T, P, C) -> (B*T, P, C)
# 2. Permute: (B*T, P, C) -> (B*T, C, P)
# 3. PointNet: (B*T, C, P) -> (B*T, 1024)
# 4. MLP: (B*T, 1024) -> (B*T, embed_dim)
# 5. Concat: 与 robot_state_obs 拼接
# 6. Reshape: (B*T, embed_dim+10) -> (B, T*(embed_dim+10))

# ==================== 输出 ====================
nx: torch.Tensor  # (B, T*(embed_dim+y_dim))
```

### 5.3 ConditionalUnet1D (流匹配网络)

```python
# 来自 diffusion_policy 库

# ==================== 输入 ====================
sample: torch.Tensor           # (B, n_pred_steps, y_dim)  带噪声的动作
 timestep: torch.Tensor         # (B,) 或 (B, 1)  时间步
 global_cond: torch.Tensor      # (B, n_obs_steps*x_dim)  条件信息

# ==================== 输出 ====================
pred_velocity: torch.Tensor    # (B, n_pred_steps, y_dim)  预测速度

# ==================== 网络结构 ====================
# U-Net 1D架构:
# - 输入投影
# - 时间步嵌入 (如果 time_conditioning=True)
# - 下采样路径: [256, 512, 1024]
# - 上采样路径
# - 输出投影
```

---

## 6. 训练流程详解

### 6.1 完整训练流程图

```
┌────────────────────────────────────────────────────────────────┐
│ Step 1: 数据准备                                                 │
├────────────────────────────────────────────────────────────────┤
│ 1.1 收集演示数据                                                  │
│     └─ bash bash/collect_data.sh                                │
│     └─ 生成 zarr 文件在 demos/sim/<task_name>/train/             │
│                                                                  │
│ 1.2 创建Dataset                                                   │
│     └─ RobotDatasetPcd 读取 zarr                                │
│     └─ SequenceSampler 采样时间序列                              │
│     └─ 数据增强 (augment_pcd_data)                               │
│                                                                  │
│ 1.3 DataLoader                                                    │
│     └─ batch_size=128                                            │
│     └─ num_workers=8                                             │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 2: 模型构建                                                 │
├────────────────────────────────────────────────────────────────┤
│ 2.1 通过 Hydra 从配置实例化                                        │
│     ├─ cfg.model: FMPolicy                                       │
│     ├─ cfg.model.obs_encoder: PointNetBackbone                  │
│     └─ cfg.model.diffusion_net: ConditionalUnet1D               │
│                                                                  │
│ 2.2 优化器: AdamW (lr=3e-5)                                      │
│ 2.3 学习率调度: Cosine warmup (5000 steps)                       │
│ 2.4 使用 EMA (指数移动平均)                                       │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 3: 训练循环 (Composer框架)                                   │
├────────────────────────────────────────────────────────────────┤
│ for epoch in range(1500):                                        │
│     for batch in dataloader:                                     │
│         ├─ composer_model.loss(batch)                            │
│         │   ├─ _norm_data(): 归一化                              │
│         │   ├─ _augment_data(): 数据增强                         │
│         │   └─ calculate_loss():                                 │
│         │       ├─ nx = obs_encoder(pcd, robot_state_obs)       │
│         │       ├─ t = sample_snr()  # 采样时间步                │
│         │       ├─ z_t = t*z_1 + (1-t)*z_0  # 插值              │
│         │       ├─ v_pred = diffusion_net(z_t, t, nx)           │
│         │       └─ loss = MSE(v_pred, z_1-z_0)                  │
│         │                                                         │
│         ├─ loss.backward()                                       │
│         └─ optimizer.step()                                      │
│                                                                  │
│     ├─ eval_dataloader: 验证集评估                               │
│     └─ save_checkpoint: 每500epoch保存                           │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 损失计算详解

```python
# 文件: pfp/policy/fm_policy.py

def calculate_loss(self, pcd, robot_state_obs, robot_state_pred):
    # 1. 编码观测
    nx = self.obs_encoder(pcd, robot_state_obs)  # (B, obs_features_dim)
    ny = robot_state_pred                        # (B, T_pred, 10)
    
    # 2. 采样时间步
    B = ny.shape[0]
    t = self._sample_snr(B)  # (B, 1, 1), 默认均匀采样 U(0,1)
    
    # 3. 初始化噪声
    z0 = self._init_noise(B)  # (B, T_pred, 10), 高斯噪声
    z1 = ny                   # 目标动作
    
    # 4. 插值 (流匹配核心)
    z_t = t * z1 + (1.0 - t) * z0   # (B, T_pred, 10)
    target_vel = z1 - z0            # (B, T_pred, 10)
    
    # 5. 网络预测
    timesteps = t.squeeze() * self.pos_emb_scale  # 时间步嵌入缩放
    pred_vel = self.diffusion_net(z_t, timesteps, global_cond=nx)
    
    # 6. 分解损失
    loss_xyz   = MSE(pred_vel[..., :3], target_vel[..., :3])
    loss_rot6d = MSE(pred_vel[..., 3:9], target_vel[..., 3:9])
    loss_grip  = MSE(pred_vel[..., 9],   target_vel[..., 9])
    
    # 7. 加权组合
    loss = l_w["xyz"] * loss_xyz + l_w["rot6d"] * loss_rot6d + l_w["grip"] * loss_grip
    # 默认权重: xyz=10.0, rot6d=10.0, grip=1.0
    
    return loss
```

### 6.3 推理详解

```python
def infer_y(self, pcd, robot_state_obs, noise=None, return_traj=False):
    # 1. 编码观测
    nx = self.obs_encoder(pcd, robot_state_obs)
    B = nx.shape[0]
    
    # 2. 初始化噪声
    z = self._init_noise(B) if noise is None else noise  # (B, T_pred, 10)
    traj = [z]
    
    # 3. 获取时间步 (支持多种调度)
    t0, dt = get_timesteps(self.flow_schedule, self.num_k_infer, exp_scale)
    # flow_schedule: "linear" | "cosine" | "exp"
    # num_k_infer: 推理步数 (默认10, 可增加到50-100)
    
    # 4. 迭代去噪
    for i in range(self.num_k_infer):
        timesteps = torch.ones((B), device=DEVICE) * t0[i]
        timesteps *= self.pos_emb_scale
        
        # 预测速度
        vel_pred = self.diffusion_net(z, timesteps, global_cond=nx)
        
        # 欧拉积分
        z = z + vel_pred * dt[i]
        traj.append(z)
    
    # 5. 返回结果
    if return_traj:
        return torch.stack(traj)  # (K, B, T_pred, 10)
    return traj[-1]  # (B, T_pred, 10)
```

---

## 7. 数据结构详解

### 7.1 数据存储格式 (Zarr)

```
demos/sim/<task_name>/
├── train/
│   └── data.zarr/
│       ├── pcd_xyz/          # 点云坐标 (N_episodes, T, P, 3)
│       ├── pcd_color/        # 点云颜色 (N_episodes, T, P, 3) uint8
│       ├── robot_state/      # 机器人状态 (N_episodes, T, 10)
│       ├── images/           # 图像 (N_episodes, T, 5, H, W, 3)
│       └── meta/
│           ├── episode_ends  # 每个episode的结束索引
│           └── ...
└── valid/
    └── data.zarr/
```

### 7.2 数据字段详解

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `pcd_xyz` | (T, ~4096, 3) | float32 | 点云XYZ坐标 |
| `pcd_color` | (T, ~4096, 3) | uint8 | 点云RGB颜色 (0-255) |
| `robot_state` | (T, 10) | float32 | 机器人状态 |
| `images` | (T, 5, 128, 128, 3) | uint8 | 5个相机图像 |

### 7.3 数据预处理流程

```python
# 文件: pfp/data/dataset_pcd.py

def __getitem__(self, idx):
    # 1. 采样序列
    sample = self.sampler.sample_sequence(idx)
    #    - sequence_length = (n_obs + n_pred) * subs_factor
    #    - 使用 SequenceSampler 处理episode边界
    
    # 2. 提取观测
    cur_step_i = self.n_obs_steps * self.subs_factor
    pcd = sample["pcd_xyz"][:cur_step_i:self.subs_factor]  # 降采样时间
    
    # 3. 可选: 添加颜色
    if self.use_pc_color:
        pcd_color = sample["pcd_color"][:cur_step_i:self.subs_factor]
        pcd_color = pcd_color.astype(np.float32) / 255.0  # 归一化到[0,1]
        pcd = np.concatenate([pcd, pcd_color], axis=-1)
    
    # 4. 提取机器人状态
    robot_state_obs  = sample["robot_state"][:cur_step_i:self.subs_factor]
    robot_state_pred = sample["robot_state"][cur_step_i::self.subs_factor]
    
    # 5. 随机采样点 (FPS或随机)
    if pcd.shape[1] > self.n_points:
        random_indices = np.random.choice(pcd.shape[1], self.n_points, replace=False)
        pcd = pcd[:, random_indices]
    
    return pcd, robot_state_obs, robot_state_pred
```

### 7.4 数据增强

```python
# 文件: pfp/data/dataset_pcd.py

def augment_pcd_data(batch):
    pcd, robot_state_obs, robot_state_pred = batch
    
    # 1. 随机SE(3)变换
    transform = pp.randn_SE3(sigma=(0.1, 0.2), device=pcd.device).matrix()  # (4,4)
    
    # 2. 应用于点云
    pcd[..., :3] = transform_th(transform, pcd[..., :3])
    
    # 3. 应用于机器人状态 (位置+旋转)
    robot_obs_pseudoposes = robot_state_obs[..., :9].reshape(*BT_robot_obs, 3, 3)
    robot_pred_pseudoposes = robot_state_pred[..., :9].reshape(*BT_robot_pred, 3, 3)
    robot_obs_pseudoposes = transform_th(transform, robot_obs_pseudoposes)
    robot_pred_pseudoposes = transform_th(transform, robot_pred_pseudoposes)
    robot_state_obs[..., :9] = robot_obs_pseudoposes.reshape(*BT_robot_obs, 9)
    robot_state_pred[..., :9] = robot_pred_pseudoposes.reshape(*BT_robot_pred, 9)
    
    # 4. 随机打乱点顺序
    idx = torch.randperm(pcd.shape[2])
    pcd = pcd[:, :, idx, :]
    
    return pcd, robot_state_obs, robot_state_pred
```

### 7.5 数据归一化

```python
# 文件: pfp/policy/fm_policy.py

def _norm_obs(self, pcd):
    """点云归一化: 仅中心化"""
    pcd[..., :3] -= torch.tensor([0.4, 0.0, 1.4], device=DEVICE)
    return pcd

def _norm_robot_state(self, robot_state):
    """机器人状态归一化"""
    robot_state[..., :3] -= torch.tensor([0.4, 0.0, 1.4], device=DEVICE)  # 位置
    robot_state[..., 9] -= torch.tensor(0.5, device=DEVICE)              # 夹爪
    return robot_state
```

---

## 8. 如何修改代码

### 8.1 添加新任务

```bash
# Step 1: 修改数据收集脚本
vim bash/collect_data.sh
# 添加新任务的两行:
# python scripts/collect_demos.py --config-name=collect_demos_train save_data=True env_config.vis=False env_config.task_name=your_new_task
# python scripts/collect_demos.py --config-name=collect_demos_valid save_data=True env_config.vis=False env_config.task_name=your_new_task

# Step 2: 收集数据
bash bash/collect_data.sh

# Step 3: 训练
python scripts/train.py task_name=your_new_test +experiment=pointflowmatch log_wandb=True
```

### 8.2 修改网络架构

```python
# 文件: conf/model/flow.yaml

diffusion_net:
  down_dims: [256, 512, 1024]  # 改为 [128, 256, 512] 减小模型
  kernel_size: 5               # 改为 3 或 7
  n_groups: 8                  # 改为 4 或 16
```

### 8.3 更换骨干网络

```bash
# 文件: conf/train.yaml

defaults:
  - model: flow
  - backbone: pointmlp  # 从 pointnet 改为 pointmlp
```

### 8.4 修改损失权重

```python
# 文件: conf/model/flow.yaml

loss_weights:
  xyz: 10.0    # 增加位置精度
  rot6d: 10.0  # 增加旋转精度
  grip: 1.0    # 增加夹爪权重 (如任务需要频繁操作夹爪)
```

### 8.5 修改推理步数

```bash
# 推理时修改
python scripts/evaluate.py policy.ckpt_name=<ckpt_name> policy.num_k_infer=50

# 或者在配置文件中修改
# 文件: conf/model/flow.yaml
num_k_infer: 50  # 默认10，增加可提高质量但减慢速度
```

### 8.6 添加自定义策略

```python
# 文件: pfp/policy/my_policy.py

from pfp.policy.fm_policy import FMPolicy

class MyPolicy(FMPolicy):
    def calculate_loss(self, pcd, robot_state_obs, robot_state_pred):
        # 自定义损失计算
        # 例如: 添加物理约束、接触力预测等
        loss = super().calculate_loss(pcd, robot_state_obs, robot_state_pred)
        
        # 添加自定义损失项
        custom_loss = self.my_custom_loss(...)
        
        return loss + custom_loss
```

### 8.7 修改数据预处理

```python
# 文件: pfp/data/dataset_pcd.py

class RobotDatasetPcd:
    def __getitem__(self, idx):
        # ... 原有代码 ...
        
        # 添加自定义预处理
        pcd = self.my_preprocess(pcd)
        
        return pcd, robot_state_obs, robot_state_pred
    
    def my_preprocess(self, pcd):
        # 例如: 法线估计、颜色直方图均衡化等
        return pcd
```

---

## 9. 常见问题

### Q1: 训练需要多少显存？

| 配置 | 显存需求 |
|------|----------|
| batch_size=128, n_points=4096 | ~10GB |
| batch_size=64, n_points=2048 | ~5GB |
| batch_size=32, n_points=1024 | ~3GB |

### Q2: 如何调试代码？

```bash
# 1. 使用可视化脚本检查数据
python scripts/vis_dataset.py

# 2. 开启rerun可视化
python scripts/evaluate.py env_runner.env_config.vis=True

# 3. 减小batch和点数快速测试
python scripts/train.py dataloader.batch_size=4 dataset.n_points=512
```

### Q3: 训练多长时间？

- 通常1500 epochs收敛
- 每个epoch约100-500 iterations (取决于数据集大小)
- RTX 3090上约需 12-24 小时

### Q4: 如何复现论文结果？

```bash
# 使用预训练权重
python scripts/evaluate.py \
    log_wandb=True \
    env_runner.env_config.vis=False \
    policy.ckpt_name=1717446544-didactic-woodpecker \
    policy.num_k_infer=50
```

---

## 附录: 关键文件速查

| 目标 | 文件 |
|------|------|
| 理解训练流程 | `scripts/train.py` |
| 理解推理流程 | `scripts/evaluate.py`, `pfp/policy/fm_policy.py` |
| 修改模型结构 | `conf/model/flow.yaml`, `pfp/policy/fm_policy.py` |
| 修改观测编码 | `pfp/backbones/pointnet.py` |
| 修改数据处理 | `pfp/data/dataset_pcd.py` |
| 修改损失函数 | `pfp/policy/fm_policy.py::calculate_loss` |
| 添加新环境 | `pfp/envs/rlbench_env.py` |
| 配置超参数 | `conf/train.yaml`, `conf/model/flow.yaml` |

---

> 最后更新: 2026-03-22
> 如有问题，请参考原始论文: [PointFlowMatch](http://pointflowmatch.cs.uni-freiburg.de/)
