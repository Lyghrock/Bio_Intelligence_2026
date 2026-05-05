from neuron import h
import numpy as np
from neuron.units import ms, mV
from matplotlib import pyplot as plt
import neuron
from pathlib import Path


_MODS_DIR = Path(__file__).resolve().parents[1] / "mods"
neuron.load_mechanisms(str(_MODS_DIR))

# 1. 建立细胞模型
soma = h.Section(name="soma")
soma.L = 10
soma.diam = 10
soma.nseg = 1
soma.Ra = 100
soma.cm = 1
soma.insert("hh1")

# 2. 插入单电极电压钳 (SEClamp)
vclamp = h.SEClamp(soma(0.5))
vclamp.rs = 0.1 # 设置极小的串联电阻，确保理想电压钳

# 3. 设置记录变量
t_vec = h.Vector().record(h._ref_t)
v_vec = h.Vector().record(soma(0.5)._ref_v)
ina_vec = h.Vector().record(soma(0.5)._ref_ina)
h_vec = h.Vector().record(soma(0.5).hh1._ref_h) # 记录失活变量 h

h.load_file('stdrun.hoc')

plt.figure(figsize=(12, 10))

# 实验 1: 改变条件阶段的时长 T0 (固定 V0)
V0_fixed = -30 * mV  
V_test = 0 * mV      
T0_list =[1, 2, 5, 10, 20, 50] 

exp1_Idata =[]
exp1_hdata =[]

for T0 in T0_list:
    vclamp.amp1 = -65; vclamp.dur1 = 5
    vclamp.amp2 = V0_fixed; vclamp.dur2 = T0
    vclamp.amp3 = V_test; vclamp.dur3 = 10
    
    h.finitialize(-65 * mV)
    h.continuerun((5 + T0 + 10) * ms)
    
    t_arr = np.array(t_vec)
    ina_arr = np.array(ina_vec)
    h_arr = np.array(h_vec)
    
    # 时间轴对齐到测试阶段 (Test step) 开始的时刻 t=0
    t_shifted = t_arr - (5 + T0)
    # h截取从 -55ms 到 10ms 的数据，看条件阶段 h 的动态变化,I_Na截取从 -1ms 到 5ms 的数据，看测试阶段的电流变化
    maskI = (t_shifted >= -1) & (t_shifted <= 5) 
    maskh = (t_shifted >= -55) & (t_shifted <= 10)
    
    exp1_Idata.append((T0, t_shifted[maskI], ina_arr[maskI]))
    exp1_hdata.append((T0, t_shifted[maskh], h_arr[maskh]))

#画 Exp 1 的 I_Na
plt.subplot(2, 2, 1)
for T0, t_m, ina_m,  in exp1_Idata:
    plt.plot(t_m, ina_m, label=r'$T_0 = {}$ ms'.format(T0))
plt.xlabel(r'Time from test step start (ms)')
plt.ylabel(r'$I_{Na}$ (mA/cm$^2$)')
plt.title(r'Exp 1: Varying duration $T_0$ ($V_0 = -30$ mV)')
plt.legend()
plt.grid(True)

#画 Exp 1 的 h (在子图1正下方)
plt.subplot(2, 2, 3)
for T0, t_m, h_m in exp1_hdata:
    plt.plot(t_m, h_m, label=r'$T_0 = {}$ ms'.format(T0))
plt.xlabel(r'Time from test step start (ms)')
plt.ylabel(r'Gating variable $h$')
plt.title(r'Exp 1: $h$ dynamics (Varying $T_0$)')
plt.legend()
plt.grid(True)

# 实验 2: 改变条件阶段的电压 V0 (固定 T0 足够长)
T0_fixed = 20 * ms   
V0_list =[-100, -80, -60, -40, -20, 0] 
V_test = 0 * mV      

exp2_Idata =[]
exp2_hdata =[]

for V0 in V0_list:
    vclamp.amp1 = -65; vclamp.dur1 = 5
    vclamp.amp2 = V0; vclamp.dur2 = T0_fixed
    vclamp.amp3 = V_test; vclamp.dur3 = 10
    
    h.finitialize(-65 * mV)
    h.continuerun((5 + T0_fixed + 10) * ms)
    
    t_arr = np.array(t_vec)
    ina_arr = np.array(ina_vec)
    h_arr = np.array(h_vec)
    
    t_shifted = t_arr - (5 + T0_fixed)
    maskI = (t_shifted >= -1) & (t_shifted <= 5)
    maskh = (t_shifted >= -25) & (t_shifted <= 10)
    
    exp2_Idata.append((V0, t_shifted[maskI], ina_arr[maskI]))
    exp2_hdata.append((V0, t_shifted[maskh], h_arr[maskh]))

#画 Exp 2 的 I_Na
plt.subplot(2, 2, 2)
for V0, t_m, ina_m in exp2_Idata:
    plt.plot(t_m, ina_m, label=r'$V_0 = {}$ mV'.format(V0))
plt.xlabel(r'Time from test step start (ms)')
plt.ylabel(r'$I_{Na}$ (mA/cm$^2$)')
plt.title(r'Exp 2: Varying voltage $V_0$ ($T_0 = 20$ ms)')
plt.legend(loc='lower right')
plt.grid(True)

#画 Exp 2 的 h (在子图2正下方)
plt.subplot(2, 2, 4)
for V0, t_m, h_m in exp2_hdata:
    plt.plot(t_m, h_m, label=r'$V_0 = {}$ mV'.format(V0))
plt.xlabel(r'Time from test step start (ms)')
plt.ylabel(r'Gating variable $h$')
plt.title(r'Exp 2: $h$ dynamics (Varying $V_0$)')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()

out_path = Path(__file__).resolve().parent / "Figure_3.png"
plt.savefig(out_path, dpi=300)
print(f"Saved figure: {out_path}")
plt.close()