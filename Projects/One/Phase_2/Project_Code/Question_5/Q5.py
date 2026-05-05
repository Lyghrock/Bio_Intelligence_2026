from neuron import h
import numpy as np
from neuron.units import ms, mV
from matplotlib import pyplot as plt
import neuron
from pathlib import Path

h.load_file('stdrun.hoc')

_MODS_DIR = Path(__file__).resolve().parents[1] / "mods"
neuron.load_mechanisms(str(_MODS_DIR))

# 1. 运行仿真的函数
def run_burst_simulation(cat_mech=None, el_val=-65):
    # 每次运行前清空空间拓扑，防止多次循环导致旧细胞残留叠加
    h('forall delete_section()') 

    soma = h.Section(name="soma")
    soma.L = 10
    soma.diam = 10
    soma.cm = 1
    soma.Ra = 100
    
    # 插入自定义的 hh1 机制
    soma.insert("hh1")
    soma(0.5).hh1.el = el_val
    soma(0.5).hh1.gnabar = 0.12
    soma(0.5).hh1.gkbar = 0.036
    soma(0.5).hh1.gl = 0.0003

    # 根据传入的机制名称决定插入哪个 T型钙通道
    if cat_mech == 'cat1g':
        soma.insert('cat1g')
        # 设置 T型钙通道最大电导和钙反转电位
        soma(0.5).cat1g.gbar = 0.01
        soma(0.5).eca = 120  # 设定钙离子反转电压
        
    # 插入电流钳触发初始去极化。改为短促的脉冲，避免强电流掩盖通道自身动力学
    iclamp = h.IClamp(soma(0.5))
    iclamp.delay = 50  # 延迟50ms，先观察一段静息态
    
    if el_val == -65:
        # 在 -65mV 时，给一个短脉冲触发 Burst
        iclamp.dur = 5     
        iclamp.amp = 0.05  
    else:
        # 在 -54.3mV 时，不给电流，观察自发放电
        iclamp.dur = 0
        iclamp.amp = 0

    # 记录变量
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)
    
    # 运行仿真
    h.finitialize(-65 * mV) 
    h.continuerun(200 * ms)
    
    return np.array(t_vec), np.array(v_vec)

# ==========================================
# 2. 批量运行仿真收集数据
# ==========================================
mechanisms = [None, 'cat1g']
labels = ['Standard HH1 (No CaT)', 'HH1 + cat1g']
colors = ['tab:blue', 'tab:orange']
linestyles = ['--', '-']

results_65 = {}
results_54 = {} # 改为测试 54.3 mV

# 循环跑完这 2 种机制
for mech in mechanisms:
    mech_key = str(mech)
    results_65[mech_key] = run_burst_simulation(cat_mech=mech, el_val=-65.0)
    results_54[mech_key] = run_burst_simulation(cat_mech=mech, el_val=-54.3)

# ==========================================
# 3. 绘图对比 (2x1 子图)
# ==========================================
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 上半部分图：el = -65 mV (短脉冲触发)
for i, mech in enumerate(mechanisms):
    mech_key = str(mech)
    t, v = results_65[mech_key]
    axs[0].plot(t, v, label=labels[i], color=colors[i], linestyle=linestyles[i], linewidth=1.5 if mech else 1.2)

axs[0].set_title("Triggered Response (el = -65 mV, 5ms Pulse at t=50ms)")
axs[0].set_ylabel("Membrane Potential (mV)")
axs[0].legend(loc='upper right')
axs[0].grid(True, linestyle='--', alpha=0.6)

# 下半部分图：el = -54.3 mV (自发放电)
for i, mech in enumerate(mechanisms):
    mech_key = str(mech)
    t, v = results_54[mech_key]
    axs[1].plot(t, v, label=labels[i], color=colors[i], linestyle=linestyles[i], linewidth=1.5 if mech else 1.2)

axs[1].set_title("Spontaneous Firing (el = -54.3 mV, No Pulse)")
axs[1].set_xlabel("Time (ms)")
axs[1].set_ylabel("Membrane Potential (mV)")
axs[1].legend(loc='upper right')
axs[1].grid(True, linestyle='--', alpha=0.6)

# 设置总体标题并保存
plt.suptitle("Impact of T-type Calcium Channels on Firing Patterns", fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig("Q5_Calcium_Bursting_cat1g.png", dpi=300)
plt.close(fig)