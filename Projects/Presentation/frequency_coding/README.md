# Neural Frequency Coding - Mini-Research Project

## 项目概述

本项目探索神经元如何作为**频率选择动力学系统**进行信息处理。

**核心假说**：生物神经元并非简单的阈值式脉冲发放装置，而是通过被动膜性质和主动离子通道实现频率选择性计算。

## 项目结构

```
frequency_coding/
├── README.md              # 本文件
├── planning.md            # 研究计划（中文）
│
├── models/                # 神经元模型
│   ├── passive_cable.py   # 被动缆线模型（实验1）
│   ├── hcn_neuron.py      # HCN共振模型（实验2）
│   ├── kcnq_neuron.py     # KCNQ模型（实验3）
│   ├── spatial_hetero.py   # 空间异质模型（实验4）
│   └── complete_filter_bank.py  # 综合滤波器组对比
│
├── experiments/           # 实验脚本（预留）
│   ├── impedance_sweep.py     # 阻抗测量实验
│   ├── signal_filtering.py    # 信号滤波实验
│   ├── kcnq_experiment.py      # KCNQ高通滤波实验
│   └── frequency_route.py     # 频率路由实验
│
└── analysis/              # 分析工具（预留）
    └── fft_tools.py       # FFT和频域分析工具

# 离子通道机制（使用官方mod）
# 位于: Projects/Presentation/reference_mod/
# - Ih.mod     # HCN/h电流通道
# - Im.mod     # KCNQ/M电流通道
# - hh1.mod    # Hodgkin-Huxley
# - pas        # 内置被动通道
```

## 使用的NEURON机制

| 机制 | 来源 | 用途 |
|------|------|------|
| `pas` | NEURON内置 | 被动膜性质 |
| `hh1` | reference_mod | Hodgkin-Huxley动作电位 |
| `Ih` | reference_mod | HCN/h电流（theta共振） |
| `Im` | reference_mod | KCNQ/M电流（高通滤波） |
| `Impedance` | NEURON内置 | 阻抗计算 |

## 依赖要求

- Python 3.8+
- NEURON (with Python support)
- NumPy
- SciPy
- Matplotlib (optional, for visualization)

## 运行方式

### 使用Conda环境（推荐）

```bash
conda run -n Bio_Intelligence python models/<model_name>.py
```

或直接使用环境中的Python：

```bash
F:\Anaconda\envs\Bio_Intelligence\python.exe models/<model_name>.py
```

### 示例：运行所有实验

```bash
# 激活conda环境后运行
cd Projects/Presentation/frequency_coding

# 实验1: 被动缆线 - 频率滤波和去噪
python models/passive_cable.py

# 实验2: HCN共振
python models/hcn_neuron.py

# 实验3: KCNQ高通滤波
python models/kcnq_neuron.py

# 实验4: 空间异质频率路由
python models/spatial_hetero.py

# 实验5: 综合滤波器组对比
python models/complete_filter_bank.py
```

## 研究框架

```
Part 1: 被动缆线 -> 低通滤波
    |-> 展示被动膜的频率依赖阻抗

Part 2: 主动通道 -> 频率选择
    |-> HCN通道 -> theta共振
    |-> KCNQ/M通道 -> 高通锐化

Part 3: 生物学意义
    |-> 听觉系统（位置编码/频率编码）
    |-> 海马theta-gamma耦合
    |-> 神经振荡与信息整合

Part 4: 真实结构中的频率协作
    |-> 空间异质性 -> 频率路由
```

## 关键发现

### 1. 被动膜的低通滤波
- 阻抗随频率增加而降低（1 Hz: ~135 MΩ -> 100 Hz: ~71 MΩ）
- 远端输入比近端输入衰减更多
- 提供时间平滑和去噪功能

### 2. HCN通道的theta共振
- 在4-8 Hz范围内产生共振峰值
- 超极化激活特性（Ih电流）
- 对海马theta节律（4-12 Hz）至关重要

### 3. KCNQ/M电流的高通滤波
- 抑制低频重复发放
- 增强脉冲时间精度
- 电压门控，阈下激活（-50至-30 mV）

### 4. 空间频率路由
- 不同树突位置偏好不同频率
- HCN富集的远端增强theta信号
- 实现单神经元的多频率处理

## 生物学意义

这些发现支持以下观点：

1. **神经元是主动的频率滤波器**，而非被动导线
2. **离子通道分布**决定了神经元的频率偏好
3. **树突形态**与通道分布共同实现频率路由
4. **神经振荡**（theta/gamma）可能被神经元选择性处理

## 参考资料

详见 `planning.md` 中的完整参考文献列表。
