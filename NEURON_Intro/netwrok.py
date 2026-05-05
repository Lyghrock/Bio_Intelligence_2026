from neuron import h

class BallAndStick:
    def __init__(self, gid):
        self._gid = gid
        self._setup_morphology()
        self._setup_biophysics()

    def _setup_morphology(self):
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.all = [self.soma, self.dend]
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157
        self.dend.L = 200
        self.dend.diam = 1

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100        # Axial resistance in Ohm * cm
            sec.cm = 1          # Membrane capacitance in micro Farads / cm^2

        self.soma.insert('hh')
        for seg in self.soma:
            seg.hh.gnabar = 0.12    # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036    # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003      # Leak conductance in S/cm2
            seg.hh.el = -54.3       # Reversal potential in mV

        # Insert passive current in the dendrite
        self.dend.insert('pas')
        for seg in self.dend:
            seg.pas.g = 0.001       # Passive conductance in S/cm2
            seg.pas.e = -65         # Leak reversal potential mV

    def __repr__(self):
        return 'BallAndStick[{}]'.format(self._gid)


source = BallAndStick(0)
target = BallAndStick(1)

# External stimulus
stim = h.NetStim()
syn_ = h.ExpSyn(source.dend(0.5))
stim.number = 1
stim.start = 9
ncstim = h.NetCon(stim, syn_)
ncstim.delay = 1
ncstim.weight[0] = 0.04

# Internal connection
syn = h.ExpSyn(target.dend(0.5))
nc = h.NetCon(source.soma(0.5)._ref_v, syn, sec=source.soma)
nc.weight[0] = 0.05
nc.delay = 5

# Record and Plot
v_s = h.Vector().record(source.soma(0.5)._ref_v)
m_s = h.Vector().record(source.soma(0.5).hh._ref_m)
n_s = h.Vector().record(source.soma(0.5).hh._ref_n)
h_s = h.Vector().record(source.soma(0.5).hh._ref_h)
v_t_soma = h.Vector().record(target.soma(0.5)._ref_v)
v_t_dend = h.Vector().record(target.dend(0.5)._ref_v)
t = h.Vector().record(h._ref_t)

h.load_file('stdrun.hoc')
h.finitialize(-65)
h.continuerun(40)

from matplotlib import pyplot
 
# Make plots export-friendly by default (dpi, tight bounding box, padding, font sizes).
pyplot.rcParams.update(
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
pyplot.figure(figsize=(8, 4))
pyplot.subplot(121)
pyplot.plot(t, v_s)
pyplot.plot(t, v_t_soma)
pyplot.plot(t, v_t_dend)
pyplot.legend(['source-soma', 'target-soma', 'target-dend'])
pyplot.xlabel('time (ms)')
pyplot.ylabel('mV')

pyplot.subplot(122)
pyplot.plot(t, m_s)
pyplot.plot(t, n_s)
pyplot.plot(t, h_s)
pyplot.legend(['m', 'n', 'h'])
pyplot.show()