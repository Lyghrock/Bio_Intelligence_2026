from neuron import h
import numpy as np
from neuron.units import ms, mV
from matplotlib import pyplot as plt
import sys
import neuron
from pathlib import Path

h.load_file('stdrun.hoc')

_MODS_DIR = Path(__file__).resolve().parents[1] / "mods"
neuron.load_mechanisms(str(_MODS_DIR))

def _run_one(*, gnabar: float, gkbar: float):
    # 每次运行前清空空间拓扑，防止多次循环导致旧细胞残留叠加
    h('forall delete_section()')

    # 1. 创建单室模型
    soma = h.Section(name="soma")
    soma.L = 10
    soma.diam = 10
    soma.cm = 1
    soma.Ra = 100

    # 2. 插入机制并设置参数（参数定义在 hh1.mod）
    soma.insert("hh1")
    soma(0.5).hh1.gnabar = gnabar
    soma(0.5).hh1.gkbar = gkbar
    soma(0.5).hh1.gl = 0.0003

    # 3. 插入 0.5 nA 强电流刺激
    iclamp = h.IClamp(soma(0.5))
    iclamp.delay = 10
    iclamp.dur = 150
    iclamp.amp = 0.5

    # 4. 记录变量
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)

    # 5. 运行仿真
    h.finitialize(-65 * mV)
    h.continuerun(200 * ms)

    # 6. 绘图
    t = np.array(t_vec)
    v = np.array(v_vec)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(t, v, color='black', linewidth=1.5)
    plt.axvspan(10, 160, color='red', alpha=0.1, label='I = 0.5 nA')

    current_temp = h.celsius
    plt.title(f'Neuron Response (gnabar={gnabar:.3f}, gkbar={gkbar:.3f}, T={current_temp}°C)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.xlim(0, 200)
    plt.ylim(-90, 60)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = "Part1_Result.png" if gnabar < 0.15 else "Part2_Result.png"
    plt.savefig(filename, dpi=300)
    print(f"Simulation complete! Plot saved as {filename}")
    plt.close(fig)


def run_testbench():
    # Part 1: 原始 HH1 参数
    _run_one(gnabar=0.12, gkbar=0.036)

    # Part 2: Fast-spiking 参数（作业中要求展示的修改）
    _run_one(gnabar=0.25, gkbar=0.10)

if __name__ == '__main__':
    run_testbench()