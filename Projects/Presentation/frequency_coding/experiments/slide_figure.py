import numpy as np
import matplotlib.pyplot as plt

# 1. 定义电压范围：从 -140 mV (极度超极化) 到 -20 mV (去极化)
v = np.linspace(-140, -20, 500)

# 防止分母为0的微小修正（对应源文件中 if v==-154.9 的处理）
v_safe = np.where(v == -154.9, v + 0.0001, v)

# 2. 计算 alpha 和 beta 速率 (单位：1/ms)
mAlpha = 0.001 * 6.43 * (v_safe + 154.9) / (np.exp((v_safe + 154.9) / 11.9) - 1)
mBeta = 0.001 * 193 * np.exp(v_safe / 33.1)

# 3. 计算稳态激活概率 (mInf) 和 时间常数 (mTau, 单位：ms)
mInf = mAlpha / (mAlpha + mBeta)
mTau = 1 / (mAlpha + mBeta)

# ==========================================
# 4. 开始绘制高颜值 PPT 专用图
# ==========================================
# 设置全局字体大小，方便 PPT 投影观看
plt.rcParams.update({'font.size': 14})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ----------------- 图 1：稳态激活曲线 (Activation Curve) -----------------
ax1.plot(v, mInf, color='#d62728', linewidth=3.5) # 使用醒目的红色
ax1.set_title('HCN Steady-State Activation ($m_{\infty}$)', fontweight='bold', pad=15)
ax1.set_xlabel('Membrane Voltage (mV)', fontweight='bold')
ax1.set_ylabel('Open Probability', fontweight='bold')
ax1.set_xlim(-140, -20)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, linestyle='--', alpha=0.6)

# 添加标注：指示“超极化激活”方向
ax1.annotate('Hyperpolarization\nActivated', xy=(-100, 0.8), xytext=(-60, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# 标出半激活电压 (V_1/2) 大约在 -80mV 左右
ax1.axhline(0.5, color='gray', linestyle=':', linewidth=2)
ax1.axvline(-82, color='gray', linestyle=':', linewidth=2) # 估算值，用于视觉指引

# ----------------- 图 2：时间常数曲线 (Time Constant) -----------------
ax2.plot(v, mTau, color='#1f77b4', linewidth=3.5) # 使用经典的蓝色
ax2.set_title('HCN Time Constant ($\\tau$)', fontweight='bold', pad=15)
ax2.set_xlabel('Membrane Voltage (mV)', fontweight='bold')
ax2.set_ylabel('Time Constant $\\tau$ (ms)', fontweight='bold')
ax2.set_xlim(-140, -20)
ax2.grid(True, linestyle='--', alpha=0.6)

# 添加标注：强调其极慢的动力学
ax2.annotate('Extremely Slow\nKinetic (>50ms)', xy=(-80, np.max(mTau)*0.8), 
             xytext=(-100, np.max(mTau)*0.9),
             fontsize=12, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="#e1f5fe", ec="#0288d1", alpha=0.9))

# 美化边框
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()

# 你可以直接显示，也可以取消下面这一行的注释来保存高清透明背景图片
# plt.savefig('hcn_kinetics.png', dpi=300, transparent=True)
plt.show()