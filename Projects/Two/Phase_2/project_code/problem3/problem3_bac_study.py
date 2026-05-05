"""Problem 3 study runner (BAC-like firing) for Project 2 Part 2&3.

Generates:
- Control vs combined traces (bAP_only / cluster_only / bAP_plus_cluster)
- Boundary exploration over timing (Δt) and synaptic strength
- A simple Ca_HVA mechanism test by scaling distal apical gCa_HVAbar_Ca_HVA

Run from this folder (so ./models and ./morphologies resolve):
    python problem3_bac_study.py --output-dir ..\\problem3_report\\figures

Notes
-----
- Requires compiled mechanisms (nrnmech.dll) in this directory.
- Uses NEURON's built-in Exp2Syn as the excitatory synapse in the scaffold.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cell import HPC, h


# -------------------------
# Global simulation defaults
# -------------------------
SPINE_DENSITY = 0.05
SEED = 100
C_TEMPERATURE = 37
V_INIT = -86
STEPS_PER_MS = 25
TSTOP = 500

# Inputs (baseline; may be overridden by scans)
SOMA_STIM_DELAY = 200
SOMA_STIM_DUR = 200
SOMA_STIM_AMP = 0.45

SYN_TAU1 = 0.3
SYN_TAU2 = 1.8
SYN_E = 0
SYN_WEIGHT_BAC = 0.0015
CLUSTER_SIZE = 20
CLUSTER_L_UM = 20

# Where to place the distal apical synaptic cluster (path distance from soma).
# Keep the assignment's default input-site location for easier tuning.
APICAL_DISTAL_TARGET_UM = 650

# Where to record distal apical Vm/cai/ica (path distance from soma).
# Note: In the provided biophys files, Ca_HVA is distributed mainly in distal
# apical segments (~685-885 um). Recording too proximally can hide Ca-dependent
# effects.
APICAL_RECORD_TARGET_UM = 750

# Distal apical region where Ca_HVA is distributed in the provided biophys files
CA_DIST_MIN_UM = 600
CA_DIST_MAX_UM = 900


@dataclass(frozen=True)
class TrialConfig:
    label: str
    use_soma_current: bool
    use_synaptic_cluster: bool
    soma_amp: float
    soma_delay: float
    soma_dur: float
    syn_weight: float
    syn_start: float
    cluster_size: int
    apical_target_um: float
    ca_hva_scale: float = 1.0


@dataclass
class PreparedModel:
    cell: HPC
    input_seg: object
    record_seg: object
    soma_clamp: object
    presyn: object | None
    synapses: list[object]
    connections: list[object]
    rec_t: object
    rec_soma_v: object
    rec_dend_v: object
    rec_dend_cai: object
    rec_dend_ica: object
    ca_hva_baseline: list[tuple[object, float]]


def _delete_all_sections() -> None:
    """Best-effort cleanup between building large models."""

    try:
        h("forall delete_section()")
    except Exception:
        pass


def _ensure_mechanisms_loaded() -> None:
    # If a key mechanism is already available, do nothing.
    # This avoids NEURON warnings like:
    #   "The user defined name already exists: Ca_HVA"
    try:
        test_sec = h.Section(name="__mech_check")
        test_sec.insert("Ca_HVA")
        return
    except Exception:
        pass

    dll_path = Path(__file__).resolve().parent / "nrnmech.dll"
    if not dll_path.exists():
        raise FileNotFoundError(
            f"Missing {dll_path}. Compile mechanisms first (e.g. run: nrnivmodl mod)."
        )

    h.nrn_load_dll(str(dll_path))


def _setup_neuron() -> None:
    # When NEURON GUI isn't imported, stdrun.hoc may not be loaded and
    # `steps_per_ms` can be undefined. Prefer a robust dt assignment.
    try:
        h.steps_per_ms = STEPS_PER_MS
        h.dt = 1.0 / h.steps_per_ms
    except Exception:
        try:
            h.load_file("stdrun.hoc")
        except Exception:
            pass
        h.dt = 1.0 / float(STEPS_PER_MS)
        try:
            h.steps_per_ms = STEPS_PER_MS
        except Exception:
            pass
    h.celsius = C_TEMPERATURE
    h.v_init = V_INIT


def count_spikes(time_ms: np.ndarray, v_mv: np.ndarray, threshold_mv: float = 0.0) -> np.ndarray:
    """Count spike times via upward threshold crossings with a short refractory."""

    crossings = (v_mv[1:] >= threshold_mv) & (v_mv[:-1] < threshold_mv)
    spike_times = time_ms[1:][crossings]
    if spike_times.size == 0:
        return spike_times

    refractory_ms = 2.0
    kept = [float(spike_times[0])]
    for t in spike_times[1:]:
        if float(t) - kept[-1] >= refractory_ms:
            kept.append(float(t))
    return np.array(kept)


def _capture_ca_hva_baseline(cell: HPC) -> list[tuple[object, float]]:
    baseline: list[tuple[object, float]] = []
    for seg in cell.segments_in_distance_range("apical", min_um=CA_DIST_MIN_UM, max_um=CA_DIST_MAX_UM):
        if hasattr(seg, "gCa_HVAbar_Ca_HVA"):
            baseline.append((seg, float(getattr(seg, "gCa_HVAbar_Ca_HVA"))))
    return baseline


def prepare_model(apical_target_um: float, cluster_size: int | None) -> PreparedModel:
    """Build a model once, then rerun many times via finitialize/continuerun."""

    cell = HPC("./", spine_density=SPINE_DENSITY)
    cell.add_full_spine(cell.HCell, 0.25, 1.35, 2.8, cell.HCell.soma[0].Ra)

    input_seg = cell.nearest_segment("apical", apical_target_um)
    record_seg = cell.nearest_segment("apical", APICAL_RECORD_TARGET_UM)

    soma_clamp = h.IClamp(cell.HCell.soma[0](0.5))
    soma_clamp.delay = SOMA_STIM_DELAY
    soma_clamp.dur = SOMA_STIM_DUR
    soma_clamp.amp = 0.0

    presyn = None
    synapses: list[object] = []
    connections: list[object] = []
    if cluster_size is not None and int(cluster_size) > 0:
        presyn = h.NetStim()
        presyn.start = SOMA_STIM_DELAY
        presyn.number = 1
        presyn.interval = 1e9
        presyn.noise = 0

        rd = h.Random(SEED)

        class ClusterConfig:
            SPINE_HEAD_X = 1
            CLUSTER_SIZE = int(cluster_size)
            CLUSTER_L = CLUSTER_L_UM

        target_segments = cell.fill_clustered_synapses_list_with_spine([input_seg], rd, ClusterConfig)
        for seg in target_segments:
            syn = h.Exp2Syn(seg.x, sec=seg.sec)
            syn.e = SYN_E
            syn.tau1 = SYN_TAU1
            syn.tau2 = SYN_TAU2
            con = h.NetCon(presyn, syn)
            con.weight[0] = 0.0
            synapses.append(syn)
            connections.append(con)

    # Record time and soma/dendrite voltage, plus a Ca proxy at the dendritic site.
    rec_t = h.Vector().record(h._ref_t)
    rec_soma_v = h.Vector().record(cell.HCell.soma[0](0.5)._ref_v)
    rec_dend_v = h.Vector().record(record_seg._ref_v)
    rec_dend_cai = h.Vector().record(record_seg._ref_cai)
    rec_dend_ica = h.Vector().record(record_seg._ref_ica)

    return PreparedModel(
        cell=cell,
        input_seg=input_seg,
        record_seg=record_seg,
        soma_clamp=soma_clamp,
        presyn=presyn,
        synapses=synapses,
        connections=connections,
        rec_t=rec_t,
        rec_soma_v=rec_soma_v,
        rec_dend_v=rec_dend_v,
        rec_dend_cai=rec_dend_cai,
        rec_dend_ica=rec_dend_ica,
        ca_hva_baseline=_capture_ca_hva_baseline(cell),
    )


def run_prepared(model: PreparedModel, cfg: TrialConfig) -> dict:
    """Run a trial by updating inputs on a pre-built model."""

    # Restore / apply Ca_HVA scaling (idempotent per run).
    for seg, base in model.ca_hva_baseline:
        setattr(seg, "gCa_HVAbar_Ca_HVA", float(base) * float(cfg.ca_hva_scale))

    n_scaled = len(model.ca_hva_baseline) if cfg.ca_hva_scale != 1.0 else 0

    # Update somatic current.
    model.soma_clamp.delay = cfg.soma_delay
    model.soma_clamp.dur = cfg.soma_dur
    model.soma_clamp.amp = float(cfg.soma_amp) if cfg.use_soma_current else 0.0

    # Update synaptic cluster.
    if model.presyn is not None:
        model.presyn.start = float(cfg.syn_start)

        w = float(cfg.syn_weight) if cfg.use_synaptic_cluster else 0.0
        for con in model.connections:
            con.weight[0] = w
    else:
        if cfg.use_synaptic_cluster:
            raise RuntimeError("Prepared model has no synaptic cluster, but cfg.use_synaptic_cluster=True")

    h.tstop = TSTOP
    h.finitialize(h.v_init)
    h.continuerun(h.tstop)

    time = np.array(model.rec_t)
    soma = np.array(model.rec_soma_v)
    dend = np.array(model.rec_dend_v)
    cai = np.array(model.rec_dend_cai)
    ica = np.array(model.rec_dend_ica)

    spikes = count_spikes(time, soma, threshold_mv=0.0)

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
        "n_scaled_ca_hva_segments": int(n_scaled),
    }


def find_clean_bac_params(apical_target_um: float) -> tuple[float, float, int, float]:
    """Find parameters where controls are clean but combined bursts.

    Goal (heuristic):
    - bAP_only: <= 1 somatic spike
    - cluster_only: 0 somatic spikes
    - combined: >= 3 somatic spikes

    Returns (soma_amp, syn_weight, cluster_size, dt_ms).
    """

    dt_ms = 5.0
    syn_start = SOMA_STIM_DELAY + dt_ms

    soma_amps = [0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36]
    cluster_sizes = [5, 10, 20, 30]
    syn_weights = [0.0004, 0.0006, 0.0008, 0.0010, 0.0012, 0.0015, 0.0018, 0.0022, 0.0028, 0.0035]

    # Precompute bAP_only spike counts for soma amps (reuse one model).
    print("[p3] tuning: scanning bAP-only soma amps...", flush=True)

    bap_ok: list[float] = []
    bap_model = prepare_model(apical_target_um=apical_target_um, cluster_size=None)
    for amp in soma_amps:
        res = run_prepared(
            bap_model,
            TrialConfig(
                label=f"bAP_only_amp={amp}",
                use_soma_current=True,
                use_synaptic_cluster=False,
                soma_amp=float(amp),
                soma_delay=SOMA_STIM_DELAY,
                soma_dur=SOMA_STIM_DUR,
                syn_weight=0.0,
                syn_start=syn_start,
                cluster_size=0,
                apical_target_um=apical_target_um,
            ),
        )
        if res["n_spikes"] <= 1:
            bap_ok.append(float(amp))

    # Drop the bAP model before building clustered-synapse models.
    bap_model = None  # type: ignore[assignment]
    _delete_all_sections()

    if not bap_ok:
        # Fallback: accept the smallest amp; later figures will still run.
        return (float(soma_amps[0]), float(syn_weights[0]), int(cluster_sizes[-1]), dt_ms)

    print(f"[p3] tuning: bAP-only acceptable amps={bap_ok}", flush=True)

    # Search for a combination where cluster_only is silent but combined bursts.
    for cluster_size in cluster_sizes:
        print(f"[p3] tuning: trying cluster_size={cluster_size}", flush=True)
        cluster_model = prepare_model(apical_target_um=apical_target_um, cluster_size=int(cluster_size))
        for syn_w in syn_weights:
            res_cluster = run_prepared(
                cluster_model,
                TrialConfig(
                    label=f"cluster_only_w={syn_w}_n={cluster_size}",
                    use_soma_current=False,
                    use_synaptic_cluster=True,
                    soma_amp=float(bap_ok[0]),
                    soma_delay=SOMA_STIM_DELAY,
                    soma_dur=SOMA_STIM_DUR,
                    syn_weight=float(syn_w),
                    syn_start=syn_start,
                    cluster_size=int(cluster_size),
                    apical_target_um=apical_target_um,
                ),
            )
            if res_cluster["n_spikes"] != 0:
                continue

            for amp in bap_ok:
                res_combined = run_prepared(
                    cluster_model,
                    TrialConfig(
                        label=f"combined_amp={amp}_w={syn_w}_n={cluster_size}",
                        use_soma_current=True,
                        use_synaptic_cluster=True,
                        soma_amp=float(amp),
                        soma_delay=SOMA_STIM_DELAY,
                        soma_dur=SOMA_STIM_DUR,
                        syn_weight=float(syn_w),
                        syn_start=syn_start,
                        cluster_size=int(cluster_size),
                        apical_target_um=apical_target_um,
                    ),
                )
                if res_combined["n_spikes"] >= 3:
                    cluster_model = None  # type: ignore[assignment]
                    _delete_all_sections()
                    return (float(amp), float(syn_w), int(cluster_size), float(dt_ms))

        cluster_model = None  # type: ignore[assignment]
        _delete_all_sections()

    # If no clean regime found, return a conservative default.
    return (float(bap_ok[0]), float(SYN_WEIGHT_BAC), int(CLUSTER_SIZE), float(dt_ms))


def plot_traces(trials: list[dict], output_path: Path, title: str) -> None:
    time = trials[0]["time_ms"]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.5), sharex=True, constrained_layout=True)
    ax_soma, ax_dend, ax_cai = axes

    for tr in trials:
        label = f"{tr['label']} (n_spikes={tr['n_spikes']})"
        ax_soma.plot(time, tr["soma_mv"], label=label)
        ax_dend.plot(time, tr["dend_mv"], label=label)
        ax_cai.plot(time, tr["dend_cai_mM"], label=label)

    ax_soma.set_title(title)
    ax_soma.set_ylabel("Soma Vm (mV)")
    ax_dend.set_ylabel("Distal apical Vm (mV)")
    ax_cai.set_ylabel("Distal apical cai (mM)")
    ax_cai.set_xlabel("Time (ms)")

    ax_soma.set_ylim(-90, 50)
    ax_dend.set_ylim(-90, 50)

    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_scan(x, y, output_path: Path, xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.8), constrained_layout=True)
    ax.plot(x, y, marker="o", ms=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path("..") / "problem3_report" / "figures"),
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()

    _ensure_mechanisms_loaded()
    _setup_neuron()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 0) Auto-tune to a "clean" BAC-like operating point
    # -------------------------------------------------
    print("[p3] Auto-tuning clean BAC-like parameters...", flush=True)
    soma_amp, syn_w, cluster_n, dt_ms = find_clean_bac_params(APICAL_DISTAL_TARGET_UM)
    syn_start = SOMA_STIM_DELAY + dt_ms

    print(
        f"[p3] Using params: soma_amp={soma_amp}, syn_w={syn_w}, cluster_n={cluster_n}, dt_ms={dt_ms}, target_um={APICAL_DISTAL_TARGET_UM}",
        flush=True,
    )

    # Build one clustered-synapse model for all subsequent experiments.
    _delete_all_sections()
    model = prepare_model(apical_target_um=APICAL_DISTAL_TARGET_UM, cluster_size=int(cluster_n))

    # -------------------------
    # 1) Core control experiment
    # -------------------------

    base_trials = [
        TrialConfig(
            label="bAP_only",
            use_soma_current=True,
            use_synaptic_cluster=False,
            soma_amp=soma_amp,
            soma_delay=SOMA_STIM_DELAY,
            soma_dur=SOMA_STIM_DUR,
            syn_weight=syn_w,
            syn_start=syn_start,
            cluster_size=cluster_n,
            apical_target_um=APICAL_DISTAL_TARGET_UM,
        ),
        TrialConfig(
            label="cluster_only",
            use_soma_current=False,
            use_synaptic_cluster=True,
            soma_amp=soma_amp,
            soma_delay=SOMA_STIM_DELAY,
            soma_dur=SOMA_STIM_DUR,
            syn_weight=syn_w,
            syn_start=syn_start,
            cluster_size=cluster_n,
            apical_target_um=APICAL_DISTAL_TARGET_UM,
        ),
        TrialConfig(
            label="bAP_plus_cluster",
            use_soma_current=True,
            use_synaptic_cluster=True,
            soma_amp=soma_amp,
            soma_delay=SOMA_STIM_DELAY,
            soma_dur=SOMA_STIM_DUR,
            syn_weight=syn_w,
            syn_start=syn_start,
            cluster_size=cluster_n,
            apical_target_um=APICAL_DISTAL_TARGET_UM,
        ),
    ]

    base_results = [run_prepared(model, cfg) for cfg in base_trials]
    plot_traces(
        base_results,
        output_dir / "p3_controls_vs_combined_traces.png",
        title=(
            f"Problem 3 controls vs combined (Δt={dt_ms} ms, syn_w={syn_w}, n={cluster_n}, target={APICAL_DISTAL_TARGET_UM} um, soma_amp={soma_amp})"
        ),
    )

    # -----------------------------
    # 2) Boundary scan: timing (Δt)
    # -----------------------------
    print("[p3] Scanning Δt boundary...", flush=True)
    delta_ts = np.arange(-40, 41, 5)
    n_spikes = []
    dend_vmax = []
    dend_cai_max = []

    for dt in delta_ts:
        res = run_prepared(
            model,
            TrialConfig(
                label=f"combined_dt={dt}",
                use_soma_current=True,
                use_synaptic_cluster=True,
                soma_amp=soma_amp,
                soma_delay=SOMA_STIM_DELAY,
                soma_dur=SOMA_STIM_DUR,
                syn_weight=syn_w,
                syn_start=SOMA_STIM_DELAY + float(dt),
                cluster_size=cluster_n,
                apical_target_um=APICAL_DISTAL_TARGET_UM,
            ),
        )
        n_spikes.append(res["n_spikes"])
        dend_vmax.append(res["dend_vmax_mv"])
        dend_cai_max.append(res["dend_cai_max_mM"])

    plot_scan(
        delta_ts,
        n_spikes,
        output_dir / "p3_boundary_delta_t_nspikes.png",
        xlabel="Δt = t_syn - t_soma (ms)",
        ylabel="Somatic spike count",
        title="Boundary scan over timing (combined condition)",
    )
    plot_scan(
        delta_ts,
        dend_cai_max,
        output_dir / "p3_boundary_delta_t_cai_max.png",
        xlabel="Δt = t_syn - t_soma (ms)",
        ylabel="Max distal apical cai (mM)",
        title="Ca proxy vs timing (combined condition)",
    )

    # --------------------------------------
    # 3) Boundary scan: synaptic strength (w)
    # --------------------------------------
    print("[p3] Scanning synaptic weight boundary...", flush=True)
    weights = np.array([0.0005, 0.0008, 0.0010, 0.0012, 0.0015, 0.0018, 0.0022])
    n_spikes_cluster_only = []
    n_spikes_combined = []

    for w in weights:
        res_cluster = run_prepared(
            model,
            TrialConfig(
                label=f"cluster_w={w}",
                use_soma_current=False,
                use_synaptic_cluster=True,
                soma_amp=soma_amp,
                soma_delay=SOMA_STIM_DELAY,
                soma_dur=SOMA_STIM_DUR,
                syn_weight=float(w),
                syn_start=syn_start,
                cluster_size=cluster_n,
                apical_target_um=APICAL_DISTAL_TARGET_UM,
            ),
        )
        res_comb = run_prepared(
            model,
            TrialConfig(
                label=f"combined_w={w}",
                use_soma_current=True,
                use_synaptic_cluster=True,
                soma_amp=soma_amp,
                soma_delay=SOMA_STIM_DELAY,
                soma_dur=SOMA_STIM_DUR,
                syn_weight=float(w),
                syn_start=syn_start,
                cluster_size=cluster_n,
                apical_target_um=APICAL_DISTAL_TARGET_UM,
            ),
        )
        n_spikes_cluster_only.append(res_cluster["n_spikes"])
        n_spikes_combined.append(res_comb["n_spikes"])

    # Plot both on the same axes for an interaction-style view.
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.8), constrained_layout=True)
    ax.plot(weights, n_spikes_cluster_only, marker="o", label="cluster_only")
    ax.plot(weights, n_spikes_combined, marker="o", label="bAP_plus_cluster")
    ax.set_title("Boundary scan over synaptic strength")
    ax.set_xlabel("Synaptic weight (a.u.)")
    ax.set_ylabel("Somatic spike count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "p3_boundary_weight_nspikes.png", dpi=200)
    plt.close(fig)

    # --------------------------------
    # 4) Mechanism test: Ca_HVA scaling
    # --------------------------------
    print("[p3] Running Ca_HVA scaling test...", flush=True)
    ca_scales = [1.0, 0.2, 0.0]
    ca_trials = []
    for s in ca_scales:
        ca_trials.append(
            run_prepared(
                model,
                TrialConfig(
                    label=f"combined_CaHVAx{s}",
                    use_soma_current=True,
                    use_synaptic_cluster=True,
                    soma_amp=soma_amp,
                    soma_delay=SOMA_STIM_DELAY,
                    soma_dur=SOMA_STIM_DUR,
                    syn_weight=syn_w,
                    syn_start=syn_start,
                    cluster_size=cluster_n,
                    apical_target_um=APICAL_DISTAL_TARGET_UM,
                    ca_hva_scale=float(s),
                ),
            )
        )

    plot_traces(
        ca_trials,
        output_dir / "p3_ca_hva_scaling_traces.png",
        title=f"Ca_HVA scaling test (combined condition, Δt={dt_ms} ms, syn_w={syn_w}, n={cluster_n}, soma_amp={soma_amp})",
    )

    # ------------------------
    # 5) Output sanity checking
    # ------------------------
    print("[p3] Verifying output figures...", flush=True)
    expected = [
        "p3_controls_vs_combined_traces.png",
        "p3_boundary_delta_t_nspikes.png",
        "p3_boundary_delta_t_cai_max.png",
        "p3_boundary_weight_nspikes.png",
        "p3_ca_hva_scaling_traces.png",
    ]
    missing = [name for name in expected if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing expected figures: {missing}")

    print("Generated figures:")
    for name in expected:
        path = (output_dir / name).resolve()
        print(f"- {path}")


if __name__ == "__main__":
    main()
