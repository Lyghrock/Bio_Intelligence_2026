import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置全局字体
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 14})

# ==========================================
# 图 1：神经元简图与实验架构 (用于 Slide 5)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.axis('off') # 关闭坐标轴

# 1. 画胞体 (Soma) - 一个圆
soma = patches.Circle((0, 0), radius=1.0, facecolor='#e0e0e0', edgecolor='#424242', lw=2, zorder=2)
ax1.add_patch(soma)
ax1.text(0, 0, 'Soma', ha='center', va='center', fontweight='bold', fontsize=16)

# 2. 画树突 (Dendrite) - 一个逐渐变细的多边形
# 树突从 x=0.9 开始，延伸到 x=8
dendrite = patches.Polygon([[0.9, 0.4], [8, 0.1], [8, -0.1], [0.9, -0.4]], 
                           facecolor='#e0e0e0', edgecolor='#424242', lw=2, zorder=1)
ax1.add_patch(dendrite)
ax1.text(4, -0.6, 'Dendrite (cable)', ha='center', va='center', fontsize=14, color='#424242')

# 3. 画记录电极 (Recording Electrodes)
# 胞体电极
ax1.annotate('Voltage\nRecording', xy=(0, 1), xytext=(-2, 2.5),
             arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=2, headwidth=8),
             fontsize=12, ha='center', color='#1f77b4', fontweight='bold')
# 树突电极 (Dend 0.8 处, 大约在 x=6.5)
ax1.annotate('Voltage\nRecording', xy=(6.5, 0.15), xytext=(8, 2.5),
             arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=2, headwidth=8),
             fontsize=12, ha='center', color='#1f77b4', fontweight='bold')

# 4. 画刺激电极 (Stimulation - ZAP current)
ax1.annotate('ZAP Current\nInjection', xy=(6.5, -0.15), xytext=(8, -2.5),
             arrowprops=dict(facecolor='#d62728', shrink=0.05, width=2, headwidth=8),
             fontsize=12, ha='center', color='#d62728', fontweight='bold')

# 画指示位置虚线
ax1.axvline(6.5, ymin=0.3, ymax=0.7, color='gray', linestyle='--', alpha=0.5)
ax1.text(6.5, 0.6, 'Dend 0.8', ha='center', va='bottom', fontsize=12)

ax1.set_xlim(-3, 10)
ax1.set_ylim(-3.5, 3.5)
plt.tight_layout()
fig1.savefig('Slide5_Model_Architecture.png', dpi=300, transparent=True)


# ==========================================
# 图 2：HCN 密度空间分布示意图 (用于 Slide 10)
# ==========================================
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.axis('off')

# 1. 画胞体和树突底图 (半透明)
soma2 = patches.Circle((0, 0), radius=1.0, facecolor='#f5f5f5', edgecolor='#9e9e9e', lw=2, zorder=1)
dendrite2 = patches.Polygon([[0.9, 0.4], [8, 0.1], [8, -0.1], [0.9, -0.4]], 
                            facecolor='#f5f5f5', edgecolor='#9e9e9e', lw=2, zorder=1)
ax2.add_patch(soma2)
ax2.add_patch(dendrite2)

# 2. 模拟树突上 HCN 通道的梯度分布 (用散点图表示通道密度)
# 产生从 1 到 8 的随机 x 坐标，向右边偏态分布
x_channels = np.concatenate([
    np.random.uniform(1, 3, 20),
    np.random.uniform(3, 5, 40),
    np.random.uniform(5, 7, 80),
    np.random.uniform(7, 8, 100)
])
# 限制 y 坐标在树突内部 (基于锥度简单估算)
y_channels = np.random.uniform(-0.2, 0.2, len(x_channels)) * (1 - (x_channels/10))
# 胞体上也放少量通道
x_soma = np.random.uniform(-0.8, 0.8, 15)
y_soma = np.random.uniform(-0.8, 0.8, 15)

# 绘制红色的 HCN 通道点
ax2.scatter(x_soma, y_soma, color='#d62728', s=10, zorder=3, alpha=0.7)
ax2.scatter(x_channels, y_channels, color='#d62728', s=10, zorder=3, alpha=0.7)

# 3. 画一条上方指示“密度激增”的曲线
x_curve = np.linspace(0, 8, 100)
# 用指数函数模拟 HCN 密度的增加 (真实海马CA1的特征: 近似指数增加 60 倍)
y_curve = 1.5 + 0.03 * np.exp(x_curve * 0.5) 
ax2.plot(x_curve, y_curve, color='#d62728', lw=3)
ax2.fill_between(x_curve, 1.5, y_curve, color='#d62728', alpha=0.2)

# 4. 文本与标注
ax2.text(0, 1.7, 'Soma\n(Low Density)', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#424242')
ax2.text(7.5, 3.5, 'Distal Dendrite\n(High HCN Density)', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#d62728')
ax2.annotate('', xy=(8, -1), xytext=(0, -1),
             arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax2.text(4, -1.2, 'Distance from Soma', ha='center', va='top', fontsize=12, fontstyle='italic')

ax2.set_xlim(-2, 10)
ax2.set_ylim(-2, 5)
plt.tight_layout()
fig2.savefig('Slide10_HCN_Gradient.png', dpi=300, transparent=True)

plt.show()