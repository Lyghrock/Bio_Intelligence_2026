from neuron import h
from neuron.units import ms, mV

# 创建胞体部分
soma = h.Section(name='soma')
soma.L = 20
soma.diam = 20

# 创建两个树突部分
dend = [h.Section(name='dend[%d]' % i) for i in range(2)]

# 设置树突的分段数
dend[0].nseg = 2 

# 连接树突到胞体
dend[0].connect(soma(0))  # 连接树突0到soma的0端
dend[1].connect(soma(1))  # 连接树突1到soma的1端

# 显示拓扑结构
h.topology()

# 打印每个部分的详细信息
print(soma.psection())
print(dend[0].psection())  
print(dend[1].psection())

shape=h.Shape(1)
shape.size(-500, 1000, -200, 500) 

from time import sleep
sleep(10)