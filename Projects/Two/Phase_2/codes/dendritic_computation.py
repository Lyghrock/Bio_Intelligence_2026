"""Experiment scaffold for the Dendritic Computation assignment.

Important NEURON/Python detail:
    Stimuli, synapses, NetCons, and recording vectors are kept in module-level
    lists below. Do not create them only inside short helper functions and then
    drop the references. Python garbage collection can remove NEURON objects
    that are no longer referenced, which silently removes inputs from a model.

Example commands:
    python dendritic_computation.py bap
    python dendritic_computation.py synapse
    python dendritic_computation.py bac
"""

from argparse import ArgumentParser
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cell import HPC, h


# Global references that keep NEURON objects alive through h.continuerun().
CELLS = []
CLAMPS = []
NETSTIMS = []
SYNAPSES = []
CONNECTIONS = []
RECORDINGS = []
TARGET_SEGMENTS = []


def clear_neuron_refs():
    """Release NEURON objects after traces have been copied to numpy arrays."""

    CELLS.clear()
    CLAMPS.clear()
    NETSTIMS.clear()
    SYNAPSES.clear()
    CONNECTIONS.clear()
    RECORDINGS.clear()
    TARGET_SEGMENTS.clear()
    gc.collect()


# Shared simulation parameters.
SPINE_DENSITY = 0.05
SEED = 100
C_TEMPERATURE = 37
V_INIT = -86
STEPS_PER_MS = 25


# Somatic current injection used in bAP and BAC examples.
SOMA_STIM_DELAY = 200
SOMA_STIM_DUR = 200
SOMA_STIM_AMP = 0.45


# Excitatory Exp2Syn parameters used in the synaptic-input scaffold.
SYN_START = 150
SYN_START_BAC = 205
SYN_WEIGHT = 0.00073
SYN_WEIGHT_BAC = 0.0015
SYN_TAU1 = 0.3
SYN_TAU2 = 1.8
SYN_E = 0
CLUSTER_L_UM = 20


# Assignment-level choices that students are expected to vary.
PROXIMAL_NA_MAX_UM = 150
APICAL_BAP_TARGET_UM = 350
APICAL_DISTAL_TARGET_UM = 650
BASAL_TARGET_UM = 120
APICAL_PROXIMAL_TARGET_UM = 120
CLUSTER_SIZES = [1, 5, 10, 20]


def plot_soma_and_dendrite(time, traces_by_condition, output_path, title, dendrite_label):
    """Plot soma and dendrite/input-site voltage on separate axes."""

    fig, (ax_soma, ax_dend) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )
    for condition, traces in traces_by_condition.items():
        ax_soma.plot(time, traces["soma"], label=condition)
        ax_dend.plot(time, traces[dendrite_label], label=condition)

    ax_soma.set_title(title)
    ax_soma.set_ylabel("Soma Vm (mV)")
    ax_dend.set_ylabel(f"{dendrite_label} Vm (mV)")
    ax_dend.set_xlabel("Time (ms)")
    for ax in (ax_soma, ax_dend):
        ax.set_ylim(-90, 50)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


parser = ArgumentParser(description=__doc__)
parser.add_argument(
    "task",
    choices=["bap", "synapse", "bac"],
    help="Which assignment scaffold to run.",
)
parser.add_argument(
    "--output-dir",
    default="figures",
    help="Directory for generated figures. Default: figures",
)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

h.steps_per_ms = STEPS_PER_MS
h.dt = 1.0 / h.steps_per_ms
h.celsius = C_TEMPERATURE
h.v_init = V_INIT
np.random.seed(SEED)


if args.task == "bap":
    # Task i: bAPs only.
    h.tstop = 500
    traces_by_condition = {}
    time = None

    for label, remove_proximal_na in [
        ("control", False),
        ("proximal_apical_Na_removed", True),
    ]:
        cell = HPC("./", spine_density=SPINE_DENSITY)
        CELLS.append(cell)
        cell.add_full_spine(cell.HCell, 0.25, 1.35, 2.8, cell.HCell.soma[0].Ra)

        apical_seg = cell.nearest_segment("apical", APICAL_BAP_TARGET_UM)
        TARGET_SEGMENTS.append(apical_seg)

        if remove_proximal_na:
            changed = cell.set_channel_density(
                "apical",
                "gNaTa_tbar_NaTa_t",
                0,
                min_um=0,
                max_um=PROXIMAL_NA_MAX_UM,
            )
            print(f"{label}: changed {changed} proximal apical segments")

        soma_clamp = h.IClamp(cell.HCell.soma[0](0.5))
        soma_clamp.delay = SOMA_STIM_DELAY
        soma_clamp.dur = SOMA_STIM_DUR
        soma_clamp.amp = SOMA_STIM_AMP
        CLAMPS.append(soma_clamp)

        t_vec = h.Vector().record(h._ref_t)
        soma_v = h.Vector().record(cell.HCell.soma[0](0.5)._ref_v)
        apic_v = h.Vector().record(apical_seg._ref_v)
        RECORDINGS.extend([t_vec, soma_v, apic_v])

        h.finitialize(h.v_init)
        h.continuerun(h.tstop)

        time = np.array(t_vec)
        traces_by_condition[label] = {
            "soma": np.array(soma_v),
            "apical_oblique": np.array(apic_v),
        }
        clear_neuron_refs()

    plot_soma_and_dendrite(
        time,
        traces_by_condition,
        output_dir / "task_i_bap_proximal_na.png",
        f"Task i: bAPs near {APICAL_BAP_TARGET_UM} um and proximal apical NaTa_t",
        "apical_oblique",
    )


if args.task == "synapse":
    # Task ii: synaptic inputs only.
    h.tstop = 350
    site_specs = [
        ("soma", "soma", False),
        ("basal_shaft", "basal", False),
        ("basal_spine", "basal", True),
        ("apical_proximal_shaft", "apical_proximal", False),
        ("apical_proximal_spine", "apical_proximal", True),
        ("apical_distal_shaft", "apical_distal", False),
        ("apical_distal_spine", "apical_distal", True),
    ]

    for figure_name, target_name, use_spines in site_specs:
        traces_by_condition = {}
        time = None

        for n_synapses in CLUSTER_SIZES:
            cell = HPC("./", spine_density=SPINE_DENSITY)
            CELLS.append(cell)
            cell.add_full_spine(cell.HCell, 0.25, 1.35, 2.8, cell.HCell.soma[0].Ra)

            if target_name == "soma":
                center_seg = cell.HCell.soma[0](0.5)
            elif target_name == "basal":
                center_seg = cell.nearest_segment("basal", BASAL_TARGET_UM)
            elif target_name == "apical_proximal":
                center_seg = cell.nearest_segment("apical", APICAL_PROXIMAL_TARGET_UM)
            else:
                center_seg = cell.nearest_segment("apical", APICAL_DISTAL_TARGET_UM)

            TARGET_SEGMENTS.append(center_seg)

            presyn = h.NetStim()
            presyn.start = SYN_START
            presyn.number = 1
            presyn.interval = 1e9
            presyn.noise = 0
            NETSTIMS.append(presyn)

            if n_synapses == 1:
                if use_spines:
                    class SingleSynConfig:
                        SPINE_HEAD_X = 1

                    target_segments = cell.fill_synapse_list_with_spine(
                        [center_seg], SingleSynConfig
                    )
                else:
                    target_segments = [center_seg]
            else:
                if use_spines:
                    rd = h.Random(SEED)

                    class ClusterConfig:
                        SPINE_HEAD_X = 1
                        CLUSTER_SIZE = n_synapses
                        CLUSTER_L = CLUSTER_L_UM

                    target_segments = cell.fill_clustered_synapses_list_with_spine(
                        [center_seg], rd, ClusterConfig
                    )
                else:
                    target_segments = []
                    half_width = CLUSTER_L_UM / (2 * center_seg.sec.L)
                    x_min = max(0, center_seg.x - half_width)
                    x_max = min(1, center_seg.x + half_width)
                    for j in range(n_synapses):
                        frac = 0.5 if n_synapses == 1 else j / max(1, n_synapses - 1)
                        x = x_min + frac * (x_max - x_min)
                        target_segments.append(center_seg.sec(x))

            TARGET_SEGMENTS.extend(target_segments)

            for seg in target_segments:
                syn = h.Exp2Syn(seg.x, sec=seg.sec)
                syn.e = SYN_E
                syn.tau1 = SYN_TAU1
                syn.tau2 = SYN_TAU2
                con = h.NetCon(presyn, syn)
                con.weight[0] = SYN_WEIGHT
                SYNAPSES.append(syn)
                CONNECTIONS.append(con)

            # Record the same anatomical center site for all n. Synapses may be
            # placed on nearby spine heads, but recording target_segments[0]
            # would compare a spine head for n=1 with a dendritic shaft or a
            # different spine for n>1, producing artificial pre-input offsets.
            record_seg = center_seg
            t_vec = h.Vector().record(h._ref_t)
            soma_v = h.Vector().record(cell.HCell.soma[0](0.5)._ref_v)
            input_v = h.Vector().record(record_seg._ref_v)
            RECORDINGS.extend([t_vec, soma_v, input_v])

            h.finitialize(h.v_init)
            h.continuerun(h.tstop)

            time = np.array(t_vec)
            traces_by_condition[f"n={n_synapses}"] = {
                "soma": np.array(soma_v),
                "input_site": np.array(input_v),
            }
            clear_neuron_refs()

        plot_soma_and_dendrite(
            time,
            traces_by_condition,
            output_dir / f"task_ii_synapse_{figure_name}.png",
            f"Task ii: synaptic input at {figure_name}",
            "input_site",
        )


if args.task == "bac":
    # Task iii: bAPs + clustered synaptic inputs.
    h.tstop = 500
    traces_by_condition = {}
    time = None

    for label, use_soma_current, use_synaptic_cluster in [
        ("bAP_only", True, False),
        ("cluster_only", False, True),
        ("bAP_plus_cluster", True, True),
    ]:
        cell = HPC("./", spine_density=SPINE_DENSITY)
        CELLS.append(cell)
        cell.add_full_spine(cell.HCell, 0.25, 1.35, 2.8, cell.HCell.soma[0].Ra)

        center_seg = cell.nearest_segment("apical", APICAL_DISTAL_TARGET_UM)
        TARGET_SEGMENTS.append(center_seg)
        record_seg = center_seg

        if use_soma_current:
            soma_clamp = h.IClamp(cell.HCell.soma[0](0.5))
            soma_clamp.delay = SOMA_STIM_DELAY
            soma_clamp.dur = SOMA_STIM_DUR
            soma_clamp.amp = SOMA_STIM_AMP
            CLAMPS.append(soma_clamp)

        if use_synaptic_cluster:
            presyn = h.NetStim()
            presyn.start = SYN_START_BAC
            presyn.number = 1
            presyn.interval = 1e9
            presyn.noise = 0
            NETSTIMS.append(presyn)

            rd = h.Random(SEED)

            class ClusterConfig:
                SPINE_HEAD_X = 1
                CLUSTER_SIZE = 20
                CLUSTER_L = CLUSTER_L_UM

            target_segments = cell.fill_clustered_synapses_list_with_spine(
                [center_seg], rd, ClusterConfig
            )
            TARGET_SEGMENTS.extend(target_segments)
            # Keep the recorded site identical across bAP_only, cluster_only,
            # and combined conditions. The synapses are clustered around this
            # center segment, but the plotted "input_site" trace is the center
            # dendritic shaft voltage.
            record_seg = center_seg

            for seg in target_segments:
                syn = h.Exp2Syn(seg.x, sec=seg.sec)
                syn.e = SYN_E
                syn.tau1 = SYN_TAU1
                syn.tau2 = SYN_TAU2
                con = h.NetCon(presyn, syn)
                con.weight[0] = SYN_WEIGHT_BAC
                SYNAPSES.append(syn)
                CONNECTIONS.append(con)

        t_vec = h.Vector().record(h._ref_t)
        soma_v = h.Vector().record(cell.HCell.soma[0](0.5)._ref_v)
        input_v = h.Vector().record(record_seg._ref_v)
        RECORDINGS.extend([t_vec, soma_v, input_v])

        h.finitialize(h.v_init)
        h.continuerun(h.tstop)

        time = np.array(t_vec)
        traces_by_condition[label] = {
            "soma": np.array(soma_v),
            "input_site": np.array(input_v),
        }
        clear_neuron_refs()

    plot_soma_and_dendrite(
        time,
        traces_by_condition,
        output_dir / "task_iii_bac.png",
        "Task iii: bAPs + clustered synaptic inputs",
        "input_site",
    )


print(f"Saved figures to {output_dir.resolve()}")
