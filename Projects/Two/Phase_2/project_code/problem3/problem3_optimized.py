"""Optimized Problem 3 BAC Firing Study for Project 2 Part 2&3.

Key design: Build cell ONCE, then run multiple trials by just changing
IClamp.amp, NetCon weights, and NetStim.start, and calling finitialize/continuerun.
This is 5-10x faster than rebuilding the cell each time.

Build:  cd codes && python problem3_optimized.py
Output: ../figures/*.png
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "codes"))
from cell import HPC, h


# -------------------------------------------------
# Simulation defaults
# -------------------------------------------------
SPINE_DENSITY = 0.05
SEED = 100
C_TEMPERATURE = 37
V_INIT = -86
STEPS_PER_MS = 25
TSTOP = 500

# Somatic current injection parameters
SOMA_STIM_DELAY = 200.0
SOMA_STIM_DUR = 200.0

# Synaptic cluster parameters
SYN_TAU1 = 0.3
SYN_TAU2 = 1.8
SYN_E = 0.0
CLUSTER_L_UM = 20.0

# Target apical distance
APICAL_TARGET_UM = 650.0
APICAL_RECORD_UM = 700.0  # record Ca/ica slightly distal to input


# -------------------------------------------------
# Trial config
# -------------------------------------------------
@dataclass(frozen=True)
class TrialConfig:
    label: str
    soma_amp: float       # 0 = no soma current
    syn_weight: float     # 0 = no synaptic input
    syn_start: float      # ms, when NetStim fires
    ca_hva_scale: float = 1.0


# -------------------------------------------------
# Count somatic spikes
# -------------------------------------------------
def count_spikes(t_ms: np.ndarray, v_mv: np.ndarray, threshold_mv: float = 0.0) -> np.ndarray:
    crossings = (v_mv[1:] >= threshold_mv) & (v_mv[:-1] < threshold_mv)
    spike_times = t_ms[1:][crossings]
    if spike_times.size == 0:
        return spike_times
    refractory_ms = 2.0
    kept = [float(spike_times[0])]
    for tt in spike_times[1:]:
        if float(tt) - kept[-1] >= refractory_ms:
            kept.append(float(tt))
    return np.array(kept)


# -------------------------------------------------
# Prepare model ONCE (build cell, spines, synapses, recording)
# -------------------------------------------------
def prepare_model(apical_target_um: float, cluster_n: int) -> tuple:
    """Returns (cell, center_seg, record_seg, soma_clamp, presyn,
               synapses, connections, ca_hva_baseline, vectors)."""

    cell = HPC(str(Path(__file__).resolve().parent.parent / "codes"), spine_density=SPINE_DENSITY)
    cell.add_full_spine(cell.HCell, 0.25, 1.35, 2.8, cell.HCell.soma[0].Ra)

    center_seg = cell.nearest_segment("apical", apical_target_um)
    record_seg = cell.nearest_segment("apical", APICAL_RECORD_UM)

    # Somatic current clamp (always present, amplitude controlled per trial)
    soma_clamp = h.IClamp(cell.HCell.soma[0](0.5))
    soma_clamp.delay = SOMA_STIM_DELAY
    soma_clamp.dur = SOMA_STIM_DUR
    soma_clamp.amp = 0.0  # controlled per trial

    # Synaptic cluster (always present, weight controlled per trial)
    presyn = h.NetStim()
    presyn.number = 1
    presyn.interval = 1e9
    presyn.noise = 0

    rd = h.Random(SEED)

    class Cfg:
        SPINE_HEAD_X = 1
        CLUSTER_SIZE = cluster_n
        CLUSTER_L = CLUSTER_L_UM

    target_segs = cell.fill_clustered_synapses_list_with_spine([center_seg], rd, Cfg)

    synapses = []
    connections = []
    for seg in target_segs:
        syn = h.Exp2Syn(seg.x, sec=seg.sec)
        syn.e = SYN_E
        syn.tau1 = SYN_TAU1
        syn.tau2 = SYN_TAU2
        con = h.NetCon(presyn, syn)
        con.weight[0] = 0.0  # controlled per trial
        synapses.append(syn)
        connections.append(con)

    # Recording vectors
    rec_t = h.Vector().record(h._ref_t)
    rec_soma = h.Vector().record(cell.HCell.soma[0](0.5)._ref_v)
    rec_dend = h.Vector().record(record_seg._ref_v)
    rec_cai = h.Vector().record(record_seg._ref_cai)
    rec_ica = h.Vector().record(record_seg._ref_ica)

    # Ca_HVA baseline (for mechanism test)
    ca_hva_baseline = []
    for seg in cell.segments_in_distance_range("apical", min_um=600, max_um=900):
        if hasattr(seg, "gCa_HVAbar_Ca_HVA"):
            ca_hva_baseline.append((seg, float(seg.gCa_HVAbar_Ca_HVA)))

    return (
        cell, center_seg, record_seg, soma_clamp, presyn,
        synapses, connections, ca_hva_baseline,
        rec_t, rec_soma, rec_dend, rec_cai, rec_ica,
    )


# -------------------------------------------------
# Run a trial (just update params, reinit, run)
# -------------------------------------------------
def run_trial(
    model, cfg: TrialConfig
) -> dict:
    (
        cell, center_seg, record_seg, soma_clamp, presyn,
        synapses, connections, ca_hva_baseline,
        rec_t, rec_soma, rec_dend, rec_cai, rec_ica,
    ) = model

    # Apply Ca_HVA scaling
    for seg, base in ca_hva_baseline:
        seg.gCa_HVAbar_Ca_HVA = base * cfg.ca_hva_scale

    # Update soma current
    soma_clamp.amp = cfg.soma_amp

    # Update synaptic cluster timing and weight
    presyn.start = cfg.syn_start
    for con in connections:
        con.weight[0] = cfg.syn_weight

    # Run simulation
    h.tstop = TSTOP
    h.finitialize(h.v_init)
    h.continuerun(h.tstop)

    time = np.array(rec_t)
    soma = np.array(rec_soma)
    dend = np.array(rec_dend)
    cai = np.array(rec_cai)
    ica = np.array(rec_ica)

    spikes = count_spikes(time, soma)

    return {
        "label": cfg.label,
        "time_ms": time,
        "soma_mv": soma,
        "dend_mv": dend,
        "dend_cai_mM": cai,
        "dend_ica_mAcm2": ica,
        "n_spikes": int(spikes.size),
        "spike_times_ms": spikes,
        "dend_vmax_mv": float(np.max(dend)),
        "dend_cai_max_mM": float(np.max(cai)),
    }


# -------------------------------------------------
# Parameter search for clean BAC regime
# -------------------------------------------------
def find_clean_bac_regime(model) -> tuple[float, float]:
    """Find (soma_amp, syn_weight) where:
    - bAP_only: <= 1 spike
    - cluster_only: 0 spikes
    - combined: >= 3 spikes

    Returns best (soma_amp, syn_weight) found.
    """

    print("[P3] Scanning for clean BAC regime...", flush=True)

    # First: find soma_amp that gives 1-2 spikes
    soma_amp_candidates = [0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35]
    good_soma_amps = []
    for amp in soma_amp_candidates:
        res = run_trial(model, TrialConfig("test_bAP", soma_amp=amp, syn_weight=0.0, syn_start=200.0))
        n = res["n_spikes"]
        print(f"  soma_amp={amp:.2f}: {n} spikes", flush=True)
        if 1 <= n <= 2:
            good_soma_amps.append(amp)

    if not good_soma_amps:
        print("  WARNING: No soma_amp gives 1-2 spikes, using 0.30")
        good_soma_amps = [0.30]

    # Second: for each soma_amp, find syn_weight that gives silent cluster but burst combined
    best = None
    best_combined = 0

    for soma_amp in good_soma_amps:
        print(f"  Trying soma_amp={soma_amp:.2f}...", flush=True)
        syn_weights = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006]

        for sw in syn_weights:
            res_cluster = run_trial(
                model,
                TrialConfig("c", soma_amp=0.0, syn_weight=sw, syn_start=SOMA_STIM_DELAY + 5.0)
            )
            if res_cluster["n_spikes"] > 0:
                continue  # cluster_only should be silent

            res_combined = run_trial(
                model,
                TrialConfig("cb", soma_amp=soma_amp, syn_weight=sw, syn_start=SOMA_STIM_DELAY + 5.0)
            )
            n_comb = res_combined["n_spikes"]
            print(f"    sw={sw:.4f}: cluster=0, combined={n_comb}", flush=True)
            if n_comb >= best_combined:
                best_combined = n_comb
                best = (soma_amp, sw)

            if n_comb >= 3:
                print(f"  FOUND: soma_amp={soma_amp}, syn_weight={sw}, combined={n_comb}")
                return (soma_amp, sw)

    if best is None:
        print("  WARNING: Could not find clean regime, using fallback")
        return (0.30, 0.003)

    print(f"  Best found: soma_amp={best[0]}, syn_weight={best[1]}, combined={best_combined}")
    return best


# -------------------------------------------------
# Plotting helpers
# -------------------------------------------------
def plot_traces(trials: list[dict], output_path: Path, title: str) -> None:
    time = trials[0]["time_ms"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    ax_soma, ax_dend, ax_cai, ax_ica = axes

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, tr in enumerate(trials):
        color = colors[i % len(colors)]
        label = f"{tr['label']} (n={tr['n_spikes']})"
        ax_soma.plot(time, tr["soma_mv"], label=label, color=color)
        ax_dend.plot(time, tr["dend_mv"], label=label, color=color)
        ax_cai.plot(time, tr["dend_cai_mM"] * 1000, label=label, color=color)  # convert to uM
        ax_ica.plot(time, tr["dend_ica_mAcm2"], label=label, color=color)

    ax_soma.set_title(title)
    ax_soma.set_ylabel("Soma Vm (mV)")
    ax_dend.set_ylabel("Distal apical Vm (mV)")
    ax_cai.set_ylabel("Distal apical\n[Ca2+]i (uM)")
    ax_ica.set_ylabel("Distal apical\nica (mA/cm2)")
    ax_cai.set_xlabel("Time (ms)")

    ax_soma.set_ylim(-90, 60)
    ax_dend.set_ylim(-90, 60)

    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_scan(x_vals, y_vals, output_path: Path, xlabel: str, ylabel: str, title: str,
              hlines: list = None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5), constrained_layout=True)
    ax.plot(x_vals, y_vals, marker="o", ms=4, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if hlines:
        for yval, ls in hlines:
            ax.axhline(yval, linestyle=ls, color="gray", alpha=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_interaction_scan(
    x_vals: np.ndarray,
    y_cluster: list,
    y_combined: list,
    output_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5), constrained_layout=True)
    ax.plot(x_vals, y_cluster, marker="o", ms=4, label="cluster_only", color="#ff7f0e")
    ax.plot(x_vals, y_combined, marker="s", ms=4, label="bAP + cluster", color="#2ca02c")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# -------------------------------------------------
# Main experiment
# -------------------------------------------------
def main():
    output_dir = Path(__file__).resolve().parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # NEURON setup
    h.load_file("stdrun.hoc")
    h.steps_per_ms = STEPS_PER_MS
    h.dt = 1.0 / h.steps_per_ms
    h.celsius = C_TEMPERATURE
    h.v_init = V_INIT

    print("[P3] Building cell model (one-time, reused for all trials)...", flush=True)
    model = prepare_model(APICAL_TARGET_UM, cluster_n=20)

    # ---- Parameter search ----
    soma_amp, syn_weight = find_clean_bac_regime(model)

    # Optimal delta_t (syn_start offset from soma delay)
    # Soma fires ~5-15ms after stim onset; cluster should arrive slightly after
    delta_t = 5.0  # ms
    syn_start = SOMA_STIM_DELAY + delta_t

    print(f"\n[P3] Running experiments with soma_amp={soma_amp}, syn_weight={syn_weight}, syn_start={syn_start}", flush=True)

    # ====================================================
    # EXP 1: Core control traces
    # ====================================================
    print("\n[P3] Exp 1: Control traces...", flush=True)
    exp1_trials = [
        TrialConfig("bAP_only", soma_amp=soma_amp, syn_weight=0.0, syn_start=syn_start),
        TrialConfig("cluster_only", soma_amp=0.0, syn_weight=syn_weight, syn_start=syn_start),
        TrialConfig("bAP_plus_cluster", soma_amp=soma_amp, syn_weight=syn_weight, syn_start=syn_start),
    ]
    exp1_results = [run_trial(model, cfg) for cfg in exp1_trials]
    for r in exp1_results:
        print(f"  {r['label']}: spikes={r['n_spikes']}, dend_vmax={r['dend_vmax_mv']:.1f}mV, dend_cai_max={r['dend_cai_max_mM']*1000:.2f}uM")

    plot_traces(
        exp1_results,
        output_dir / "p3_exp1_control_traces.png",
        title=(
            f"Problem 3: bAP + clustered synaptic input interaction\n"
            f"(soma_amp={soma_amp}, syn_w={syn_weight}, delta_t={delta_t}ms, apical={APICAL_TARGET_UM}um)"
        ),
    )

    # ====================================================
    # EXP 2: delta_t boundary scan
    # ====================================================
    print("\n[P3] Exp 2: delta_t boundary scan...", flush=True)
    delta_ts = np.arange(-30, 31, 5, dtype=float)
    n_spikes_combined = []
    dend_vmax_combined = []
    cai_max_combined = []
    n_spikes_cluster = []
    dend_vmax_cluster = []

    for dt in delta_ts:
        start = SOMA_STIM_DELAY + dt
        res_comb = run_trial(
            model,
            TrialConfig(f"dt={dt}", soma_amp=soma_amp, syn_weight=syn_weight, syn_start=start)
        )
        res_clu = run_trial(
            model,
            TrialConfig(f"clu_dt={dt}", soma_amp=0.0, syn_weight=syn_weight, syn_start=start)
        )
        n_spikes_combined.append(res_comb["n_spikes"])
        dend_vmax_combined.append(res_comb["dend_vmax_mv"])
        cai_max_combined.append(res_comb["dend_cai_max_mM"])
        n_spikes_cluster.append(res_clu["n_spikes"])
        dend_vmax_cluster.append(res_clu["dend_vmax_mv"])

    plot_interaction_scan(
        delta_ts,
        n_spikes_cluster,
        n_spikes_combined,
        output_dir / "p3_exp2_delta_t_spikes.png",
        xlabel="delta_t = t_syn - t_soma (ms)",
        ylabel="Somatic spike count",
        title="Spike count vs timing delta_t (cluster_only vs combined)",
    )
    plot_scan(
        delta_ts,
        np.array(cai_max_combined) * 1000,
        output_dir / "p3_exp2_delta_t_cai.png",
        xlabel="delta_t = t_syn - t_soma (ms)",
        ylabel="Peak [Ca2+]i (uM)",
        title="Dendritic calcium vs timing delta_t (combined condition)",
    )
    plot_scan(
        delta_ts,
        dend_vmax_combined,
        output_dir / "p3_exp2_delta_t_dendv.png",
        xlabel="delta_t = t_syn - t_soma (ms)",
        ylabel="Peak dendritic Vm (mV)",
        title="Dendritic voltage vs timing delta_t (combined condition)",
    )

    # ====================================================
    # EXP 3: Synaptic weight boundary scan
    # ====================================================
    print("\n[P3] Exp 3: Synaptic weight boundary scan...", flush=True)
    weights = np.array([0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.007])
    n_spikes_clu_w = []
    n_spikes_comb_w = []
    dend_vmax_clu_w = []
    dend_vmax_comb_w = []

    for w in weights:
        res_clu = run_trial(
            model,
            TrialConfig(f"clu_w={w}", soma_amp=0.0, syn_weight=float(w), syn_start=syn_start)
        )
        res_comb = run_trial(
            model,
            TrialConfig(f"comb_w={w}", soma_amp=soma_amp, syn_weight=float(w), syn_start=syn_start)
        )
        n_spikes_clu_w.append(res_clu["n_spikes"])
        n_spikes_comb_w.append(res_comb["n_spikes"])
        dend_vmax_clu_w.append(res_clu["dend_vmax_mv"])
        dend_vmax_comb_w.append(res_comb["dend_vmax_mv"])

    plot_interaction_scan(
        weights * 1000,  # convert to 10^-3 units for readability
        n_spikes_clu_w,
        n_spikes_comb_w,
        output_dir / "p3_exp3_weight_spikes.png",
        xlabel="Synaptic weight (x10^-3)",
        ylabel="Somatic spike count",
        title="Spike count vs synaptic weight (cluster_only vs combined)",
    )
    plot_scan(
        weights * 1000,
        np.array(dend_vmax_comb_w),
        output_dir / "p3_exp3_weight_dendv.png",
        xlabel="Synaptic weight (x10^-3)",
        ylabel="Peak dendritic Vm (mV)",
        title="Dendritic voltage vs synaptic weight (combined)",
    )

    # ====================================================
    # EXP 4: Ca_HVA mechanism test
    # ====================================================
    print("\n[P3] Exp 4: Ca_HVA mechanism test...", flush=True)
    ca_scales = [1.0, 0.5, 0.1, 0.0]
    exp4_results = []
    for s in ca_scales:
        res = run_trial(
            model,
            TrialConfig(f"combined_CaHVA_{s}", soma_amp=soma_amp, syn_weight=syn_weight,
                       syn_start=syn_start, ca_hva_scale=float(s))
        )
        exp4_results.append(res)
        print(f"  Ca_HVA x{s}: spikes={res['n_spikes']}, dend_vmax={res['dend_vmax_mv']:.1f}mV, dend_cai_max={res['dend_cai_max_mM']*1000:.2f}uM")

    plot_traces(
        exp4_results,
        output_dir / "p3_exp4_ca_hva_mechanism.png",
        title=f"Ca_HVA mechanism test (soma_amp={soma_amp}, syn_w={syn_weight}, delta_t={delta_t}ms)",
    )

    # ====================================================
    # EXP 5: Soma spike count vs timing (show coincidence window)
    # ====================================================
    print("\n[P3] Exp 5: Coincidence window (soma amp scan)...", flush=True)
    # Pick the best soma_amp found and also test slightly weaker
    soma_amps_test = [soma_amp, soma_amp - 0.02]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5), constrained_layout=True)
    colors_amp = ["#2ca02c", "#17becf"]
    for i, s_amp in enumerate(soma_amps_test):
        n_sp = []
        for dt in delta_ts:
            res = run_trial(
                model,
                TrialConfig(f"", soma_amp=s_amp, syn_weight=syn_weight, syn_start=SOMA_STIM_DELAY + dt)
            )
            n_sp.append(res["n_spikes"])
        ax.plot(delta_ts, n_sp, marker="o", ms=4, label=f"soma_amp={s_amp}", color=colors_amp[i])
    ax.set_title("Coincidence detection window: spike count vs delta_t")
    ax.set_xlabel("delta_t = t_syn - t_soma (ms)")
    ax.set_ylabel("Somatic spike count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "p3_exp5_coincidence_window.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'p3_exp5_coincidence_window.png'}")

    # ====================================================
    # EXP 6: Dendritic voltage comparison (zoom around stimulus)
    # ====================================================
    print("\n[P3] Exp 6: Dendritic voltage zoom...", flush=True)
    # Use exp1 results for this
    time = exp1_results[0]["time_ms"]
    t_start_idx = np.searchsorted(time, SOMA_STIM_DELAY - 20)
    t_end_idx = np.searchsorted(time, SOMA_STIM_DELAY + 150)

    fig, ax = plt.subplots(1, 1, figsize=(9, 3.5), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["bAP_only", "cluster_only", "bAP+cluster"]
    for i, res in enumerate(exp1_results):
        ax.plot(
            time[t_start_idx:t_end_idx],
            res["dend_mv"][t_start_idx:t_end_idx],
            label=labels[i],
            color=colors[i],
        )
    ax.set_title("Dendritic voltage zoomed around stimulus (distal apical)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Distal apical Vm (mV)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "p3_exp6_dend_zoom.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'p3_exp6_dend_zoom.png'}")

    # ====================================================
    # Verify all figures
    # ====================================================
    print("\n[P3] Verification...", flush=True)
    expected = [
        "p3_exp1_control_traces.png",
        "p3_exp2_delta_t_spikes.png",
        "p3_exp2_delta_t_cai.png",
        "p3_exp2_delta_t_dendv.png",
        "p3_exp3_weight_spikes.png",
        "p3_exp3_weight_dendv.png",
        "p3_exp4_ca_hva_mechanism.png",
        "p3_exp5_coincidence_window.png",
        "p3_exp6_dend_zoom.png",
    ]
    missing = [n for n in expected if not (output_dir / n).exists()]
    if missing:
        print(f"  MISSING figures: {missing}")
    else:
        print(f"  All {len(expected)} figures saved to {output_dir}")
        for n in expected:
            print(f"    - {n}")

    print("\n[P3] Done!", flush=True)


if __name__ == "__main__":
    main()
