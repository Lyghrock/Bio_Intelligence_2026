from neuron import h

mt = h.MechanismType(0)

for i in range(int(mt.count())):
    mt.select(i)
    print(mt.selected())
    