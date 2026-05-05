from neuron import h
from neuron.units import ms,mV
from matplotlib import pyplot as plt

# Make plots export-friendly by default (dpi, tight bounding box, padding, font sizes).
plt.rcParams.update(
	{
		"figure.dpi": 120,
		"savefig.dpi": 300,
		"savefig.bbox": "tight",
		"savefig.pad_inches": 0.1,
		"savefig.facecolor": "white",
		"savefig.transparent": False,
		"font.size": 11,
		"axes.labelsize": 11,
		"xtick.labelsize": 10,
		"ytick.labelsize": 10,
		"legend.fontsize": 10,
	}
)
soma =h.Section(name="soma")
soma.L = 20
soma.diam = 20
soma.insert("hh")
iclamp = h.IClamp(soma(0.5))
iclamp.delay =2
iclamp.dur =0.1
iclamp.amp =0.9
v=h.Vector().record(soma(0.5)._ref_v)
t= h.Vector().record(h._ref_t)
h.load_file('stdrun.hoc')
h.finitialize(-65 * mV)
h.continuerun(40 * ms)
plt.figure(figsize=(8,4))
plt.plot(t,v)
plt.xlabel("time(ms)")
plt.ylabel('mv')
plt.show()