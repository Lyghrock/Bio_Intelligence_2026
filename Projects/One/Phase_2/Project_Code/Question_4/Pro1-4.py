from neuron import h
import numpy as np
from neuron.units import ms, mV
from matplotlib import pyplot as plt
import neuron
from pathlib import Path


_MODS_DIR = Path(__file__).resolve().parents[1] / "mods"
neuron.load_mechanisms(str(_MODS_DIR))

#建立细胞模型
soma = h.Section(name="soma")
soma.L = 10
soma.diam = 10
soma.nseg = 1
soma.Ra = 100
soma.cm = 1
soma.insert("hh1")

# 设置记录变量
t_vec = h.Vector().record(h._ref_t)
v_vec = h.Vector().record(soma(0.5)._ref_v)
h_vec = h.Vector().record(soma(0.5).hh1._ref_h)

h.load_file('stdrun.hoc')

plt.figure(figsize=(12, 10))

# 协议 A (Protocol A)
# I0 = 0.1 nA, delay=20, dur=20
iclamp_A = h.IClamp(soma(0.5))
iclamp_A.delay = 20 * ms
iclamp_A.dur = 20 * ms
iclamp_A.amp = 0.1

h.finitialize(-65 * mV)
h.continuerun(60 * ms)

plt.subplot(3, 2, 1)
plt.plot(t_vec, v_vec, color='blue')
plt.title('Protocol A: Voltage Response')
plt.ylabel(r'$V_m$ (mV)')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(t_vec, h_vec, color='red')
plt.title(r'Protocol A: Inactivation Variable $h$')
plt.ylabel(r'Gating variable $h$')
plt.grid(True)

iclamp_A = None # 运行完销毁刺激器

# 协议 B (Protocol B)
# I1(1.0nA) 在 10~20ms; I0(0.1nA) 在 20~40ms
iclamp_B1 = h.IClamp(soma(0.5))
iclamp_B1.delay = 10 * ms; iclamp_B1.dur = 10 * ms; iclamp_B1.amp = 1.0

iclamp_B0 = h.IClamp(soma(0.5))
iclamp_B0.delay = 20 * ms; iclamp_B0.dur = 20 * ms; iclamp_B0.amp = 0.1

h.finitialize(-65 * mV)
h.continuerun(60 * ms)

plt.subplot(3, 2, 3)
plt.plot(t_vec, v_vec, color='blue')
plt.title('Protocol B: Voltage Response')
plt.ylabel(r'$V_m$ (mV)')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(t_vec, h_vec, color='red')
plt.title(r'Protocol B: Inactivation Variable $h$')
plt.ylabel(r'Gating variable $h$')
plt.grid(True)

iclamp_B1 = None; iclamp_B0 = None

# 协议 C (Protocol C)
# I1(1.0nA) 在 10~20ms; I2(-1.0nA) 在 20~21ms;  I0(0.1nA) 在 21~41ms
iclamp_C1 = h.IClamp(soma(0.5))
iclamp_C1.delay = 10 * ms; iclamp_C1.dur = 10 * ms; iclamp_C1.amp = 1.0

iclamp_C2 = h.IClamp(soma(0.5))
iclamp_C2.delay = 20 * ms; iclamp_C2.dur = 1 * ms; iclamp_C2.amp = -1.0

iclamp_C0 = h.IClamp(soma(0.5))
iclamp_C0.delay = 21 * ms; iclamp_C0.dur = 20 * ms; iclamp_C0.amp = 0.1

h.finitialize(-65 * mV)
h.continuerun(60 * ms)

plt.subplot(3, 2, 5)
plt.plot(t_vec, v_vec, color='blue')
plt.title('Protocol C: Voltage Response')
plt.xlabel('Time (ms)')
plt.ylabel(r'$V_m$ (mV)')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(t_vec, h_vec, color='red')
plt.title(r'Protocol C: Inactivation Variable $h$')
plt.xlabel('Time (ms)')
plt.ylabel(r'Gating variable $h$')
plt.grid(True)

plt.tight_layout()

out_path = Path(__file__).resolve().parent / "Figure_4.png"
plt.savefig(out_path, dpi=300)
print(f"Saved figure: {out_path}")
plt.close()