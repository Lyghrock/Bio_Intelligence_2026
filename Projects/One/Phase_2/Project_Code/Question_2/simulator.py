from __future__ import annotations

from pathlib import Path
import importlib.util
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

try:
	from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
	def tqdm(iterable, *args, **kwargs):
		return iterable


def make_iv_equilibria_plot(
	*,
	v_rest_mV: float = -65.0,
	iclamp_amp_nA: float = 0.05,
	pulse_dur_ms: float = 0.5,
	v_min_mV: float = -100.0,
	v_max_mV: float = 60.0,
	num: int = 801,
	out_path: Path,
):
	"""Plot the approximate fast-subsystem I–V curve and mark (un)stable equilibria.

	We use the Q2 hint approximation for brief stimuli:
	- freeze the slow gates at rest: h=h(Vrest), n=n(Vrest)
	- let m follow its steady-state value at each V: m=m_inf(V)

	Equilibria satisfy: I_ion(V) = I_inj_density.
	Stability in the 1D reduced dynamics C dV/dt = I_inj - I_ion(V):
	- stable if dI_ion/dV > 0 at the intersection
	- unstable if dI_ion/dV < 0 at the intersection
	"""
	# Use the analytic HH rate formulas (from hh1.mod) for a clean I–V plot.
	# Parameters are the same defaults used in Q1 (hh1.mod):
	gnabar = 0.12
	gkbar = 0.036
	gl = 0.0003
	ena = 50.0
	ek = -77.5
	el = -54.3

	def vtrap(x: np.ndarray, y: float) -> np.ndarray:
		x_over_y = x / y
		out = np.empty_like(x_over_y, dtype=float)
		small = np.abs(x_over_y) < 1e-6
		out[small] = y * (1 - x_over_y[small] / 2)
		out[~small] = x[~small] / (np.exp(x_over_y[~small]) - 1)
		return out

	def minf(v: np.ndarray) -> np.ndarray:
		alpha = 0.1 * vtrap(-(v + 40.0), 10.0)
		beta = 4.0 * np.exp(-(v + 65.0) / 18.0)
		sum_ = alpha + beta
		return alpha / sum_

	def hinf(v: np.ndarray) -> np.ndarray:
		alpha = 0.07 * np.exp(-(v + 65.0) / 20.0)
		beta = 1.0 / (np.exp(-(v + 35.0) / 10.0) + 1.0)
		sum_ = alpha + beta
		return alpha / sum_

	def ninf(v: np.ndarray) -> np.ndarray:
		alpha = 0.01 * vtrap(-(v + 55.0), 10.0)
		beta = 0.125 * np.exp(-(v + 65.0) / 80.0)
		sum_ = alpha + beta
		return alpha / sum_

	def tau_m_ms(v: np.ndarray) -> np.ndarray:
		alpha = 0.1 * vtrap(-(v + 40.0), 10.0)
		beta = 4.0 * np.exp(-(v + 65.0) / 18.0)
		return 1.0 / (alpha + beta)

	V = np.linspace(float(v_min_mV), float(v_max_mV), int(num), dtype=float)

	# Frozen slow gates at rest.
	v_rest_arr = np.asarray([float(v_rest_mV)], dtype=float)
	h_rest = float(hinf(v_rest_arr)[0])
	n_rest = float(ninf(v_rest_arr)[0])

	# Finite-duration approximation for m during a pulse of length T:
	# m(T) = m_rest + (m_inf(V) - m_rest) * (1 - exp(-T/tau_m(V)))
	m_rest = float(minf(v_rest_arr)[0])
	m_inf = minf(V)
	tau = tau_m_ms(V)
	T = float(pulse_dur_ms)
	m_eff = m_rest + (m_inf - m_rest) * (1.0 - np.exp(-T / tau))

	Iion = (
		gnabar * (m_eff**3) * h_rest * (V - ena)
		+ gkbar * (n_rest**4) * (V - ek)
		+ gl * (V - el)
	)

	# Current density corresponding to IClamp amplitude (geometry matches Q1: L=diam=10 µm).
	area_um2 = np.pi * 10.0 * 10.0
	area_cm2 = area_um2 * 1e-8
	Iinj_mA_cm2 = (float(iclamp_amp_nA) * 1e-6) / area_cm2  # nA -> mA, then /cm^2

	# Find intersections Iion(V) - Iinj = 0 via sign changes.
	diff = Iion - Iinj_mA_cm2
	sgn = np.sign(diff)
	idx = np.flatnonzero(sgn[:-1] * sgn[1:] <= 0)
	roots = []
	for j in idx:
		v0, v1 = V[j], V[j + 1]
		d0, d1 = diff[j], diff[j + 1]
		if d0 == d1:
			continue
		vr = v0 + (0.0 - d0) * (v1 - v0) / (d1 - d0)
		roots.append(float(vr))

	# Classify stability using dIion/dV at the root.
	dI_dV = np.gradient(Iion, V)
	stable_v = []
	unstable_v = []
	for vr in roots:
		slope = float(np.interp(vr, V, dI_dV))
		if slope > 0:
			stable_v.append(vr)
		else:
			unstable_v.append(vr)

	fig, ax = plt.subplots(figsize=(7.5, 5.0))
	ax.plot(V, Iion, color="C0", linewidth=2.0, label=r"$I_{ion}(V)$ (freeze $h,n$ at rest)")
	ax.axhline(Iinj_mA_cm2, color="C3", linestyle="--", linewidth=1.8, label=rf"$I_{{inj}}$ density (amp={iclamp_amp_nA:g} nA)")

	if stable_v:
		ax.scatter(stable_v, [Iinj_mA_cm2] * len(stable_v), s=70, color="C2", edgecolor="white", zorder=5, label="stable eq.")
	if unstable_v:
		ax.scatter(unstable_v, [Iinj_mA_cm2] * len(unstable_v), s=70, color="C3", marker="x", zorder=6, label="unstable eq.")

	ax.set_title(f"Approximate I–V curve and equilibria (Q2, pulse {pulse_dur_ms:g} ms)")
	ax.set_xlabel("V (mV)")
	ax.set_ylabel(r"$I_{ion}$ (mA/cm$^2$)")
	ax.grid(True, alpha=0.25)
	ax.legend(loc="best")
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=250)
	plt.close(fig)

	return {
		"out_path": out_path,
		"Iinj_mA_cm2": float(Iinj_mA_cm2),
		"stable_V_mV": stable_v,
		"unstable_V_mV": unstable_v,
		"v_rest_mV": float(v_rest_mV),
	}


def make_3d_summary(
	*,
	q1_simulator,
	title: str,
	param_name: str,
	param_values: list[float],
	fixed_kwargs: dict,
	out_path: Path,
):
	"""Generate a 2x2 figure where each subplot is a 3D axis (t vs param vs signal)."""
	records = []
	for p in tqdm(param_values, desc=f"3D sweep: {param_name}"):
		kwargs = dict(fixed_kwargs)
		kwargs[param_name] = p
		_, data = q1_simulator(plot=False, return_data=True, verbose=False, **kwargs)
		records.append(data)

	t = np.asarray(records[0]["t_ms"], dtype=float)
	p = np.asarray(param_values, dtype=float)
	T, P = np.meshgrid(t, p)

	V = np.asarray([r["v_mV"] for r in records], dtype=float)
	ina = np.asarray([r["ina_mA_cm2"] for r in records], dtype=float)
	ik = np.asarray([r["ik_mA_cm2"] for r in records], dtype=float)
	m = np.asarray([r["m"] for r in records], dtype=float)
	n = np.asarray([r["n"] for r in records], dtype=float)
	hh = np.asarray([r["h"] for r in records], dtype=float)
	gna = np.asarray([r["gna_S_cm2"] for r in records], dtype=float)
	gk = np.asarray([r["gk_S_cm2"] for r in records], dtype=float)

	def surf(ax, z: np.ndarray, *, label: str, color: str):
		ax.plot_surface(T, P, z, alpha=0.65, linewidth=0, antialiased=True, color=color)
		ax.plot([], [], [], color=color, label=label)

	fig = plt.figure(figsize=(14, 10))
	fig.suptitle(title)

	ax_v = fig.add_subplot(2, 2, 1, projection="3d")
	ax_i = fig.add_subplot(2, 2, 2, projection="3d")
	ax_g = fig.add_subplot(2, 2, 3, projection="3d")
	ax_cond = fig.add_subplot(2, 2, 4, projection="3d")

	surf(ax_v, V, label="V (mV)", color="C0")
	ax_v.set_xlabel("t (ms)")
	ax_v.set_ylabel(param_name)
	ax_v.set_zlabel("V (mV)")
	ax_v.legend(loc="best")

	surf(ax_i, ina, label="ina (mA/cm2)", color="C4")
	surf(ax_i, ik, label="ik (mA/cm2)", color="C5")
	ax_i.set_xlabel("t (ms)")
	ax_i.set_ylabel(param_name)
	ax_i.set_zlabel("Iion (mA/cm2)")
	ax_i.legend(loc="best")

	surf(ax_g, m, label="m", color="C1")
	surf(ax_g, n, label="n", color="C2")
	surf(ax_g, hh, label="h", color="C3")
	ax_g.set_xlabel("t (ms)")
	ax_g.set_ylabel(param_name)
	ax_g.set_zlabel("gating (0-1)")
	ax_g.legend(loc="best")

	surf(ax_cond, gna, label="gna (S/cm2)", color="C7")
	surf(ax_cond, gk, label="gk (S/cm2)", color="C8")
	ax_cond.set_xlabel("t (ms)")
	ax_cond.set_ylabel(param_name)
	ax_cond.set_zlabel("conductance (S/cm2)")
	ax_cond.legend(loc="best")

	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	plt.close(fig)
	return out_path


def _count_spikes(*, t_ms: np.ndarray, v_mV: np.ndarray, stim_start_ms: float, threshold_mV: float = 0.0) -> int:
	"""Count spikes by upward threshold crossings after stimulation begins."""
	t_ms = np.asarray(t_ms, dtype=float)
	v_mV = np.asarray(v_mV, dtype=float)
	mask = t_ms >= stim_start_ms
	if not np.any(mask):
		return 0
	t = t_ms[mask]
	v = v_mV[mask]
	crossings = np.flatnonzero((v[:-1] < threshold_mV) & (v[1:] >= threshold_mV))
	if crossings.size == 0:
		return 0
	# De-duplicate crossings that can happen around the same spike peak
	min_isi_ms = 1.0
	spike_times = t[crossings]
	count = 1
	last_t = spike_times[0]
	for tt in spike_times[1:]:
		if tt - last_t >= min_isi_ms:
			count += 1
			last_t = tt
	return int(count)


def make_phase_boundary(
	*,
	q1_simulator,
	amp_values: list[float],
	dur_values: list[float],
	iclamp_delay: float,
	dt: float,
	tstop: float,
	out_path: Path,
):
	"""Create an amp–dur phase diagram and extract the threshold boundary curve."""
	amp_values = sorted(float(a) for a in amp_values)
	dur_values = sorted(float(d) for d in dur_values)

	spike_grid = np.zeros((len(dur_values), len(amp_values)), dtype=int)
	boundary_amp = np.full((len(dur_values),), np.nan, dtype=float)

	for di, dur in enumerate(tqdm(dur_values, desc="Boundary sweep over dur")):
		first_spike_amp = None
		for ai, amp in enumerate(amp_values):
			_, data = q1_simulator(
				plot=False,
				return_data=True,
				verbose=False,
				iclamp_amp=amp,
				iclamp_delay=iclamp_delay,
				iclamp_dur=dur,
				dt=dt,
				tstop=tstop,
			)
			spikes = _count_spikes(
				t_ms=np.asarray(data["t_ms"], dtype=float),
				v_mV=np.asarray(data["v_mV"], dtype=float),
				stim_start_ms=float(iclamp_delay),
				threshold_mV=0.0,
			)
			spike_grid[di, ai] = 1 if spikes > 0 else 0
			if first_spike_amp is None and spikes > 0:
				first_spike_amp = amp
				# keep going to fill grid (phase plot), but we can skip if speed matters
		if first_spike_amp is not None:
			boundary_amp[di] = float(first_spike_amp)

	fig, ax = plt.subplots(figsize=(8.5, 5.5))
	# Use imshow for robust categorical phase diagram (0/1)
	im = ax.imshow(
		spike_grid.T,
		origin="lower",
		aspect="auto",
		cmap="Greys",
		extent=[min(dur_values), max(dur_values), min(amp_values), max(amp_values)],
		vmin=0,
		vmax=1,
	)
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label("spike (0/1)")

	valid = np.isfinite(boundary_amp)
	if np.any(valid):
		ax.plot(np.asarray(dur_values)[valid], boundary_amp[valid], color="C0", linewidth=2.5, label="threshold boundary")
		ax.scatter(np.asarray(dur_values)[valid], boundary_amp[valid], s=18, color="C0")

	ax.set_title("Phase diagram (amp vs dur) with spike threshold boundary")
	ax.set_xlabel("duration (ms)")
	ax.set_ylabel("amplitude (nA)")
	ax.legend(loc="best")
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=200)
	plt.close(fig)
	return {
		"out_path": out_path,
		"dur_values": dur_values,
		"boundary_amp": boundary_amp.tolist(),
	}


def run_sweep(
	*,
	amp_values: list[float],
	dur_values: list[float],
	iclamp_delay: float = 1.0,
	dt: float = 0.025,
	tstop: float = 20.0,
):
	"""Run a fine-ish sweep over (amp, dur), saving 2D plots for every pair.

	2D plots go into Question_2/fig/.
	3D summary plots go into Question_2/ (same level as this file).
	"""
	# Load Q1 simulator with a direct file import (no packages needed)
	q2_dir = Path(__file__).resolve().parent
	q1_sim_path = q2_dir.parent / "Question_1" / "simulator.py"
	if not q1_sim_path.exists():
		raise FileNotFoundError(f"Cannot find Q1 simulator at: {q1_sim_path}")
	spec = importlib.util.spec_from_file_location("q1_simulator", q1_sim_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to create import spec for: {q1_sim_path}")
	q1_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(q1_module)
	q1_simulator = q1_module.simulator

	fmt = lambda v: f"{v:.4f}".rstrip("0").rstrip(".")
	fig_dir = q2_dir / "fig"
	fig_dir.mkdir(parents=True, exist_ok=True)

	# 1) Save per-parameter 2D figures (Q1 style)
	total_2d = len(amp_values) * len(dur_values)
	for amp, dur in tqdm(product(amp_values, dur_values), total=total_2d, desc="2D sweep (amp×dur)"):
		fname = f"V_I_g_cond__amp_{fmt(amp)}nA__dur_{fmt(dur)}ms.png"
		out = fig_dir / fname
		if out.exists():
			continue
		q1_simulator(
			save_path=out,
			show=False,
			plot=True,
			return_data=False,
			verbose=False,
			iclamp_amp=amp,
			iclamp_delay=iclamp_delay,
			iclamp_dur=dur,
			dt=dt,
			tstop=tstop,
		)

	# 2) 3D summaries (t vs param vs signal)
	#    Since 3D only supports one "parameter axis", we create two summaries:
	#    - sweep amplitude at fixed duration
	#    - sweep duration at fixed amplitude
	baseline_amp = 0.05
	baseline_dur = 0.5

	amp_sweep_path = q2_dir / "summary_3d_sweep_amp.png"
	dur_sweep_path = q2_dir / "summary_3d_sweep_dur.png"

	if not amp_sweep_path.exists():
		make_3d_summary(
			q1_simulator=q1_simulator,
			title=f"3D summary: sweep iclamp_amp (fixed iclamp_dur={baseline_dur} ms)",
			param_name="iclamp_amp",
			param_values=amp_values,
			fixed_kwargs={
				"iclamp_amp": baseline_amp,  # overwritten by param loop
				"iclamp_dur": baseline_dur,
				"iclamp_delay": iclamp_delay,
				"dt": dt,
				"tstop": tstop,
			},
			out_path=amp_sweep_path,
		)

	# 3) Phase diagram + boundary curve
	boundary_path = q2_dir / "phase_boundary_amp_dur.png"
	if not boundary_path.exists():
		make_phase_boundary(
			q1_simulator=q1_simulator,
			amp_values=amp_values,
			dur_values=dur_values,
			iclamp_delay=iclamp_delay,
			dt=dt,
			tstop=tstop,
			out_path=boundary_path,
		)

	if not dur_sweep_path.exists():
		make_3d_summary(
			q1_simulator=q1_simulator,
			title=f"3D summary: sweep iclamp_dur (fixed iclamp_amp={baseline_amp} nA)",
			param_name="iclamp_dur",
			param_values=dur_values,
			fixed_kwargs={
				"iclamp_amp": baseline_amp,
				"iclamp_dur": baseline_dur,  # overwritten by param loop
				"iclamp_delay": iclamp_delay,
				"dt": dt,
				"tstop": tstop,
			},
			out_path=dur_sweep_path,
		)

	return {
		"fig_dir": fig_dir,
		"amp_sweep_3d": amp_sweep_path,
		"dur_sweep_3d": dur_sweep_path,
		"boundary_phase": boundary_path,
		"num_2d": len(amp_values) * len(dur_values),
	}
if __name__ == "__main__":
	# Main logic (keep it here, per request)
	amp_start, amp_stop, amp_step = 0.0, 0.10, 0.005
	dur_start, dur_stop, dur_step = 0.10, 1.00, 0.05
	iclamp_delay = 1.0
	dt = 0.025
	tstop = 20.0

	if amp_step <= 0 or dur_step <= 0:
		raise ValueError("amp_step and dur_step must be > 0")

	amp_values = [float(x) for x in np.arange(amp_start, amp_stop + amp_step / 2, amp_step)]
	dur_values = [float(x) for x in np.arange(dur_start, dur_stop + dur_step / 2, dur_step)]

	result = run_sweep(
		amp_values=amp_values,
		dur_values=dur_values,
		iclamp_delay=iclamp_delay,
		dt=dt,
		tstop=tstop,
	)

	# 4) I–V curve for the fast-gate threshold intuition (cheap, single plot)
	q2_dir = Path(__file__).resolve().parent
	iv_path = q2_dir / "iv_curve_equilibria.png"
	if not iv_path.exists():
		make_iv_equilibria_plot(out_path=iv_path, iclamp_amp_nA=0.05, pulse_dur_ms=0.5, v_rest_mV=-54.3)
		print(f"Saved I–V equilibria plot to: {iv_path}")
	else:
		print(f"I–V equilibria plot already exists: {iv_path}")
	print(
		f"Saved {result['num_2d']} 2D figs to: {result['fig_dir']}\n"
		f"Saved 3D amp sweep to: {result['amp_sweep_3d']}\n"
		f"Saved 3D dur sweep to: {result['dur_sweep_3d']}"
	)
