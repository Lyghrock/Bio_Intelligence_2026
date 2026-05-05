import os
import sys

from neuron import h, gui
import numpy as np
from matplotlib import pyplot as plt
import math
from cell import *


path = './'
spine_density = 0.05


seed = 100 
MAX_JITTER = 0 #synchronous activation
cluster_type = "sync_clusters"


np.random.seed(seed)

class config_params():
    pass

config = config_params()

config.CLUSTER_TYPE = None
config.TAU_1_AMPA = 0.3
config.TAU_2_AMPA = 1.8
config.TAU_1_NMDA = 8.019 
config.TAU_2_NMDA = 34.9884
config.N_NMDA = 0.28011
config.GAMMA_NMDA = 0.0765685
config.AMPA_W = 0.00073027
config.NMDA_W = 0.00131038
config.NMDA_W_BLOCKED = 0
config.E_SYN = 0

config.Spike_time = 200
config.SPINE_HEAD_X = 1
config.CLUSTER_L = 20
"""---------- PLEASE PLAY WITH THIS PATAMETER ----------"""
config.CLUSTER_SIZE = 0
"""-----------------------------------------------------"""

h.steps_per_ms = 25
h.dt = 1.0 / h.steps_per_ms
h.celsius = 37
h.v_init = -86
h.tstop = 500


rd = h.Random(seed)

cell = HPC(path, spine_density)
cell.add_full_spine(cell.HCell, 0.25, 1.35, 2.8, cell.HCell.soma[0].Ra)

def show_length():
    for sec in h.allsec():
        print(f'Section name: {sec.name()}')
        print(f'Length: {sec.L}, Diameter: {sec.diam}')

def remove_na_channel_from_prox_dend_(cell, max_dist_um=0):
    """Optional example: remove NaTa_t channels from proximal apical dendrite.

    Proximal/distal is defined by path distance from soma. The default
    max_dist_um=0 keeps the original model unchanged. Increase it, for example
    to 150, to test how proximal dendritic sodium channels affect bAPs.
    """

    return cell.set_channel_density(
        "apical", "gNaTa_tbar_NaTa_t", 0, min_um=0, max_um=max_dist_um
    )

#show_length()
remove_na_channel_from_prox_dend_(cell)

"""---------- UNCOMMENT THE FOLLOWING CODE TO ADD CURRENT STIMULI ON SOMA ----------"""
stim_soma = h.IClamp(cell.HCell.soma[0](0.5))
stim_soma.delay = config.Spike_time
###---------- PLEASE PLAY WITH THESE PATAMETERS ----------###
stim_soma.dur = 200
stim_soma.amp = 1
"""-----------------------------------------------------"""

# presynaptic stimuli to synapse(s) on apical dendrite
Stim1 = h.NetStim()
Stim1.interval = 1e9
"""---------- PLEASE PLAY WITH THIS PATAMETER ----------"""
Stim1.start = 0
"""-----------------------------------------------------"""
Stim1.noise = 0
Stim1.number = 1

config.stim = Stim1

"""---------- UNCOMMENT THE FOLLOWING CODE TO ASSIGN SINGLE SYNAPSE ON APICAL DENDRITE ----------
synaptic_segments = cell.fill_synapse_list_with_spine([cell.HCell.apic[50](0.5)], config)
cell.add_synapses_on_list_of_segments(synaptic_segments, cell.synlist, cell.conlist, config)
configure_synaptic_delayes(MAX_JITTER, cell.conlist, rd, config, cluster_type=None)
-----------------------------------------------------"""

"""---------- UNCOMMENT THE FOLLOWING CODE TO ASSIGN CLUSTERED SYNAPSES ON APICAL DENDRITE ----------
synaptic_segments = cell.fill_clustered_synapses_list_with_spine([cell.HCell.apic[50](0.5)], rd, config)
cell.add_synapses_on_list_of_segments(synaptic_segments, cell.synlist, cell.conlist, config)
configure_synaptic_delayes(MAX_JITTER, cell.conlist, rd, config, cluster_type=None)
-----------------------------------------------------"""

#distance
print(f"distance: {h.distance(cell.HCell.soma[0](0.5), cell.HCell.apic[50](0.5))}")

#record the voltage response at soma and input sites
rec_t = h.Vector().record(h._ref_t)
rec_vsoma = h.Vector().record(cell.HCell.soma[0](0.5)._ref_v)
rec_vapic = h.Vector().record(cell.HCell.apic[50](0.5)._ref_v)

h.finitialize(h.v_init)
h.continuerun(h.tstop)

#plot and save
plt.plot(rec_t, rec_vsoma, label='soma')
plt.plot(rec_t, rec_vapic, label='apic')
plt.xlabel('Time (ms)')
plt.ylim(-90, 40)
plt.ylabel('Voltage (mV)')
plt.legend()
plt.show()
plt.savefig('voltage_response.png')

