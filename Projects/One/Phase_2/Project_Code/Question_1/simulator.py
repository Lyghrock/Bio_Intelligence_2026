from neuron import h
from neuron.units import mV, ms
import neuron
import matplotlib.pyplot as plt
from pathlib import Path


_MODS_DIR = Path(__file__).resolve().parents[1] / "mods"
neuron.load_mechanisms(str(_MODS_DIR))

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

def simulator(
	save_path: str | Path = "simulator.png",
	*,
	show: bool = False,
	plot: bool = True,
	return_data: bool = False,
	verbose: bool = False,
	iclamp_amp: float = 0.05,
	iclamp_delay: float = 1.0,
	iclamp_dur: float = 0.5,
	dt: float = 0.025,
	tstop: float = 20.0,
):
	"""Run a single-compartment HH simulation and save the resulting figure.

	Args:
		save_path: Output image path. Relative paths are resolved next to this file.
		show: If True, also display the figure window.
		plot: If True, generate and save a 2D (Q1-style) figure.
		return_data: If True, also return recorded time series data.
		verbose: If True, print extra debug info.
		iclamp_amp/delay/dur: IClamp parameters.
		dt: Simulation time step (ms).
		tstop: Simulation stop time (ms).
	"""
	# Morphology_Setup
	soma = h.Section(name="Soma_0")
	soma.nseg = 1
	soma.L = soma.diam = 10
	soma.Ra = 100
	soma.cm = 1

	# Bio-Phisical Strcuture
	soma.insert("hh1")
	if verbose:
		print(dir(soma(0.5)))
	soma(0.5).hh1.gkbar = 0.036
	soma(0.5).hh1.gnabar = 0.12
	soma(0.5).hh1.gl = 0.0003

	# Injection_Current
	clmp = h.IClamp(soma(0.5))
	clmp.amp = iclamp_amp
	clmp.delay = iclamp_delay
	clmp.dur = iclamp_dur

	# Record_Variables
	ts = h.Vector().record(h._ref_t)
	Vm_t = h.Vector().record(soma(0.5)._ref_v)
	m_t = h.Vector().record(soma(0.5).hh1._ref_m)
	n_t = h.Vector().record(soma(0.5).hh1._ref_n)
	h_t = h.Vector().record(soma(0.5).hh1._ref_h)
	ik_t = h.Vector().record(soma(0.5).k_ion._ref_ik)
	ina_t = h.Vector().record(soma(0.5).na_ion._ref_ina)
	iclamp_t = h.Vector().record(clmp._ref_i)
	gna_t = h.Vector().record(soma(0.5).hh1._ref_gna)
	gk_t = h.Vector().record(soma(0.5).hh1._ref_gk)

	# Run
	h.load_file("stdrun.hoc")
	h.dt = dt
	h.tstop = tstop
	h.finitialize(-65 * mV)
	h.continuerun(h.tstop)

	data = None
	if return_data:
		data = {
			"t_ms": list(ts),
			"v_mV": list(Vm_t),
			"m": list(m_t),
			"n": list(n_t),
			"h": list(h_t),
			"ik_mA_cm2": list(ik_t),
			"ina_mA_cm2": list(ina_t),
			"iclamp_nA": list(iclamp_t),
			"gna_S_cm2": list(gna_t),
			"gk_S_cm2": list(gk_t),
			"params": {
				"iclamp_amp": float(iclamp_amp),
				"iclamp_delay": float(iclamp_delay),
				"iclamp_dur": float(iclamp_dur),
				"dt": float(dt),
				"tstop": float(tstop),
			},
		}

	# Prevent section buildup in parameter sweeps
	h.delete_section(sec=soma)

	if not plot:
		return (None, data) if return_data else None

	# Plot
	fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12, 8))

	ax_v = axs[0, 0]
	ax_i = axs[0, 1]
	ax_g = axs[1, 0]
	ax_cond = axs[1, 1]

	# 1) V(t)
	ax_v.plot(ts, Vm_t, color="C0", label="V (mV)")
	ax_v.set_ylabel("V (mV)")
	ax_v.legend(loc="best", fontsize=9)

	# 2) Ina(t), Ik(t)
	ax_i.plot(ts, ina_t, color="C4", label="ina (mA/cm2)")
	ax_i.plot(ts, ik_t, color="C5", label="ik (mA/cm2)")
	ax_i.set_ylabel("Iion (mA/cm2)")
	ax_i.legend(loc="best", fontsize=9)

	# 3) m(t), n(t), h(t)
	ax_g.plot(ts, m_t, color="C1", linestyle="--", label="m")
	ax_g.plot(ts, n_t, color="C2", linestyle="--", label="n")
	ax_g.plot(ts, h_t, color="C3", linestyle="--", label="h")
	ax_g.set_xlabel("t (ms)")
	ax_g.set_ylabel("gating (0-1)")
	ax_g.legend(loc="best", fontsize=9)

	# 4) gna(t), gk(t)
	ax_cond.plot(ts, gna_t, color="C7", label="gna (S/cm2)")
	ax_cond.plot(ts, gk_t, color="C8", label="gk (S/cm2)")
	ax_cond.set_xlabel("t (ms)")
	ax_cond.set_ylabel("conductance (S/cm2)")
	ax_cond.legend(loc="best", fontsize=9)

	fig.tight_layout()

	# Save (export-friendly defaults are set in rcParams above)
	out_path = Path(save_path)
	if not out_path.is_absolute():
		out_path = Path(__file__).resolve().parent / out_path
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	if verbose:
		print(f"Saved figure: {out_path}")

	if show:
		plt.show()
	else:
		plt.close(fig)

	return (out_path, data) if return_data else out_path


if __name__ == "__main__":
	simulator()
