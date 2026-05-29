"""
Shared HH + HCN resonance model and analysis utilities.
"""

from neuron import h
import numpy as np
import os


h.load_file('stdrun.hoc')
h.CVode().active(0)
try:
    h.celsius = 6.3
except Exception:
    pass

_ref_mod_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'reference_mod', 'nrnmech.dll'
))
if os.path.exists(_ref_mod_path):
    h.nrn_load_dll(_ref_mod_path)
else:
    print(f"[WARN] Mechanism DLL not found at: {_ref_mod_path}")


THETA_BAND = (4.0, 10.0)
DEFAULT_G_H = 0.0008
HCN_SOMA_DENSITY_FACTOR = 0.25
HCN_DEND_DENSITY_BASE = 0.5
HCN_DEND_DENSITY_SLOPE = 0.5
DEFAULT_EH_HCN = -45.0
DEFAULT_SIM_DT = 0.5
VALIDATION_DT = 0.025
THETA_SHADE = '#fff2b2'

FREQUENCY_SWEEP_THETA = np.array(
    [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25],
    dtype=float,
)
FREQUENCY_SWEEP_ACTIVE = np.array(
    [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12,
     15, 20, 25, 30, 35, 40, 50, 60, 80, 100],
    dtype=float,
)

_hh_notice_printed = False


def has_spike(voltage, threshold=-20):
    return bool(np.any(np.asarray(voltage) > threshold))


def fit_sine_amplitude(t_ms, y, freq_hz):
    t_sec = np.asarray(t_ms) / 1000.0
    y = np.asarray(y)
    phase = 2 * np.pi * freq_hz * t_sec
    design = np.column_stack([
        np.sin(phase),
        np.cos(phase),
        np.ones_like(phase),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(np.sqrt(coeffs[0] ** 2 + coeffs[1] ** 2))


def compute_step_metrics(t, v, step_start=200, step_dur=500):
    step_end = step_start + step_dur
    rest_mask = (t >= step_start - 100) & (t < step_start)
    step_mask = (t >= step_start) & (t < step_end)
    late_mask = (t >= step_end - 50) & (t < step_end)
    v_rest = float(np.mean(v[rest_mask])) if np.any(rest_mask) else -65.0
    v_min = float(np.min(v[step_mask]))
    v_late = float(np.mean(v[late_mask]))
    return {
        'v_rest': v_rest,
        'v_min': v_min,
        'v_late': v_late,
        'hyperpolarization_depth': v_rest - v_min,
        'sag_recovery': v_late - v_min,
    }


def normalized_gain(freqs, gain):
    return gain / (np.interp(0.5, freqs, gain) + 1e-12)


def frequency_response_metrics(freqs, gain, theta_band=THETA_BAND):
    z_norm = normalized_gain(freqs, gain)
    peak_idx = int(np.argmax(z_norm))
    theta = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    if not np.any(theta):
        raise RuntimeError("Frequency sweep has no theta-band samples")
    theta_norm = z_norm[theta]
    theta_freqs = freqs[theta]
    theta_idx = int(np.argmax(theta_norm))
    return {
        'norm': z_norm,
        'ri': float(z_norm[peak_idx]),
        'peak_freq': float(freqs[peak_idx]),
        'global_ri': float(z_norm[peak_idx]),
        'global_peak_freq': float(freqs[peak_idx]),
        'theta_ri': float(theta_norm[theta_idx]),
        'theta_peak_freq': float(theta_freqs[theta_idx]),
        'z_ref': float(np.interp(0.5, freqs, gain)),
    }


def detected_resonance_band(freqs, peak_freq):
    if THETA_BAND[0] <= peak_freq <= THETA_BAND[1]:
        return THETA_BAND

    band_low = max(THETA_BAND[0], peak_freq - 10.0)
    band_high = min(float(freqs[-1]), peak_freq + 10.0)
    if band_high - band_low > 24.0:
        center = 0.5 * (band_low + band_high)
        band_low = max(THETA_BAND[0], center - 12.0)
        band_high = min(float(freqs[-1]), center + 12.0)
    if band_high <= band_low:
        band_low = max(float(freqs[0]), peak_freq - 2.0)
        band_high = min(float(freqs[-1]), peak_freq + 2.0)
    return float(band_low), float(band_high)


def resonance_band_label(resonance_band):
    if np.isclose(resonance_band[0], THETA_BAND[0]) and np.isclose(resonance_band[1], THETA_BAND[1]):
        return 'Theta band'
    return 'Detected resonance band'


def resonance_prominence(freqs, gain, resonance_band):
    z = normalized_gain(freqs, gain)
    active = (freqs >= resonance_band[0]) & (freqs <= resonance_band[1])
    low = (freqs >= 1) & (freqs < resonance_band[0])
    high = (freqs > resonance_band[1]) & (freqs <= freqs[-1])
    if not np.any(active) or not np.any(low):
        raise RuntimeError("Frequency sweep missing resonance or low-frequency flank samples")
    if not np.any(high):
        print("[WARN] Frequency range may still be too narrow to estimate high-frequency flank.")
        high = freqs == freqs[-1]
    active_peak = np.max(z[active])
    flank_mean = 0.5 * (np.mean(z[low]) + np.mean(z[high]))
    return float(active_peak - flank_mean)


def set_tight_ylim(ax, values, include_one=True, pad_fraction=0.08):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if include_one:
        values = np.concatenate([values, np.array([1.0])])
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    span = max(y_max - y_min, 0.02)
    pad = span * pad_fraction
    ax.set_ylim(y_min - pad, y_max + pad)


def plot_zap_trace(ax, t_control, v_control, t_hcn, v_hcn,
                   control_label, hcn_label, title):
    mask_control = t_control >= 200
    mask_hcn = t_hcn >= 200
    baseline_control = np.mean(v_control[(t_control >= 300) & (t_control <= 600)])
    baseline_hcn = np.mean(v_hcn[(t_hcn >= 300) & (t_hcn <= 600)])
    ax.plot(
        t_control[mask_control],
        v_control[mask_control] - baseline_control,
        color='tab:blue',
        alpha=0.65,
        lw=1.0,
        label=control_label,
    )
    ax.plot(
        t_hcn[mask_hcn],
        v_hcn[mask_hcn] - baseline_hcn,
        color='tab:red',
        lw=1.2,
        label=hcn_label,
    )
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Soma voltage fluctuation (mV)')
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(200, 4000)


class HHHCNResonanceNeuron:
    """Soma HH active membrane with optional HCN/Ih in soma and dendrite."""

    def __init__(self, name, with_hcn=False, dend_L=300, g_h=DEFAULT_G_H,
                 g_pas_dend=0.0003, eh_hcn=DEFAULT_EH_HCN,
                 hcn_soma_density=HCN_SOMA_DENSITY_FACTOR,
                 sim_dt=DEFAULT_SIM_DT):
        self.name = name
        self.with_hcn = with_hcn
        self.g_h = g_h
        self.eh_hcn = eh_hcn
        self.hcn_soma_density = hcn_soma_density
        self.sim_dt = sim_dt
        self.hh_mech = None

        self.soma = h.Section(name=f'soma_{name}')
        self.dend = h.Section(name=f'dend_{name}')

        self.soma.L = 10
        self.soma.diam = 10
        self.soma.cm = 1.0
        self.soma.Ra = 100
        self.soma.insert('pas')
        self.soma.g_pas = 0.0001
        self.soma.e_pas = -65
        self._insert_somatic_hh()

        self.dend.L = dend_L
        self.dend.diam = 2.0
        self.dend.nseg = 31
        self.dend.cm = 1.0
        self.dend.Ra = 100
        self.dend.insert('pas')
        self.dend.g_pas = g_pas_dend
        self.dend.e_pas = -65

        if with_hcn:
            self._set_hcn_reversal()
            self.soma.insert('Ih')
            for seg in self.soma:
                seg.Ih.gIhbar = g_h * hcn_soma_density
            self.dend.insert('Ih')
            for seg in self.dend:
                seg.Ih.gIhbar = g_h * (HCN_DEND_DENSITY_BASE + HCN_DEND_DENSITY_SLOPE * seg.x)

        self.dend.connect(self.soma(1))

    def _insert_somatic_hh(self):
        global _hh_notice_printed
        try:
            self.soma.insert('hh1')
            self.hh_mech = 'hh1'
        except Exception:
            try:
                self.soma.insert('hh')
                self.hh_mech = 'hh'
                print(f"[WARN] {self.name}: hh1 unavailable; using built-in hh")
            except Exception as exc:
                print(f"[WARN] Could not insert HH/HH1 into soma for {self.name}: {exc}")
                return

        for seg in self.soma:
            mech = getattr(seg, self.hh_mech)
            for attr, value in (
                ('gnabar', 0.08),
                ('gkbar', 0.024),
                ('gl', 0.0001),
                ('el', -65.0),
            ):
                try:
                    setattr(mech, attr, value)
                except Exception:
                    print(f"[WARN] Could not set {self.hh_mech}.{attr} for {self.name}; using default.")

        if not _hh_notice_printed:
            print("[OK] Added HH/HH1 Na/K conductance to soma only.")
            _hh_notice_printed = True

    def _set_hcn_reversal(self):
        try:
            h.ehcn_Ih = self.eh_hcn
        except Exception:
            print(f"[WARN] {self.name}: could not set Ih ehcn; using mechanism default")

    def _record_vectors(self):
        t_vec = h.Vector()
        v_soma = h.Vector()
        v_dend = h.Vector()
        t_vec.record(h._ref_t)
        v_soma.record(self.soma(0.5)._ref_v)
        v_dend.record(self.dend(0.8)._ref_v)
        return t_vec, v_soma, v_dend

    def run_current(self, duration, dt=None, hold_amp=0.0, wave=None,
                    ac_loc='dend', operating_v=-65.0, extra_stim=None,
                    check_spike=True):
        dt = self.sim_dt if dt is None else dt
        t_vec, v_soma, v_dend = self._record_vectors()

        hold = h.IClamp(self.soma(0.5))
        hold.delay = 0
        hold.dur = duration
        hold.amp = hold_amp

        wave_vec = None
        if wave is not None:
            stim_site = self.dend(0.8) if ac_loc == 'dend' else self.soma(0.5)
            stim = h.IClamp(stim_site)
            stim.delay = 0
            stim.dur = duration
            stim.amp = 0
            wave_vec = h.Vector(wave)
            wave_vec.play(stim._ref_amp, dt)

        if extra_stim is not None:
            extra = h.IClamp(self.soma(0.5))
            extra.delay = extra_stim['delay']
            extra.dur = extra_stim['dur']
            extra.amp = extra_stim['amp']

        if abs(float(h.dt) - dt) > 1e-12:
            h.dt = dt
        h.finitialize(operating_v)
        h.tstop = duration
        while h.t < h.tstop:
            h.fadvance()

        if wave_vec is not None:
            wave_vec.play_remove()

        t = np.array(t_vec)
        soma = np.array(v_soma)
        dend = np.array(v_dend)
        if check_spike and (has_spike(soma) or has_spike(dend)):
            raise RuntimeError(f"{self.name} became suprathreshold")
        return t, soma, dend

    def calibrate_holding_current(self, operating_v=-65.0, duration=1200,
                                  dt=None, check_spike=False):
        dt = self.sim_dt if dt is None else dt
        lo, hi = -0.6, 0.4
        best_amp = 0.0
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            t, soma, _ = self.run_current(
                duration=duration,
                dt=dt,
                hold_amp=mid,
                wave=None,
                operating_v=operating_v,
                check_spike=check_spike,
            )
            steady = float(np.mean(soma[t > duration - 200]))
            best_amp = mid
            if steady > operating_v:
                hi = mid
            else:
                lo = mid

        t, soma, dend = self.run_current(
            duration=duration,
            dt=dt,
            hold_amp=best_amp,
            wave=None,
            operating_v=operating_v,
            check_spike=check_spike,
        )
        soma_v = float(np.mean(soma[t > duration - 200]))
        dend_v = float(np.mean(dend[t > duration - 200]))
        return best_amp, soma_v, dend_v

    def single_frequency_response(self, freq, ac_amp=0.01, hold_amp=0.0,
                                  n_cycles=8, discard_cycles=3,
                                  operating_v=-65.0, ac_loc='dend'):
        dt = self.sim_dt
        duration = n_cycles * 1000.0 / freq
        t_wave = np.arange(0, duration, dt)
        wave = ac_amp * np.sin(2 * np.pi * freq * (t_wave / 1000.0))
        t, soma, dend = self.run_current(
            duration=duration,
            dt=dt,
            hold_amp=hold_amp,
            wave=wave,
            ac_loc=ac_loc,
            operating_v=operating_v,
            check_spike=True,
        )

        start = discard_cycles * 1000.0 / freq
        mask = (t >= start) & (t < duration)
        if np.count_nonzero(mask) < 10:
            raise RuntimeError(f"Not enough steady-state samples at {freq:.2f} Hz")
        soma_amp = fit_sine_amplitude(t[mask], soma[mask] - np.mean(soma[mask]), freq)
        dend_amp = fit_sine_amplitude(t[mask], dend[mask] - np.mean(dend[mask]), freq)
        return {
            'freq': freq,
            'soma_amp': soma_amp,
            'dend_amp': dend_amp,
            'soma_gain': soma_amp / ac_amp,
            'dend_gain': dend_amp / ac_amp,
        }

    def zap_trace(self, duration=4000, f_start=0.5, f_end=25, ac_amp=0.01,
                  hold_amp=0.0, ramp_ms=200.0, operating_v=-65.0,
                  ac_loc='dend'):
        dt = self.sim_dt
        t_wave = np.arange(0, duration, dt)
        t_sec = t_wave / 1000.0
        rate = (f_end - f_start) / (duration / 1000.0)
        phase = 2 * np.pi * (f_start * t_sec + 0.5 * rate * t_sec ** 2)
        ramp = np.ones_like(t_wave)
        ramp_mask = t_wave < ramp_ms
        ramp[ramp_mask] = 0.5 * (1 - np.cos(np.pi * t_wave[ramp_mask] / ramp_ms))
        wave = ac_amp * ramp * np.sin(phase)
        return self.run_current(
            duration=duration,
            dt=dt,
            hold_amp=hold_amp,
            wave=wave,
            ac_loc=ac_loc,
            operating_v=operating_v,
            check_spike=True,
        )

    def hyperpolarization_test(self, hold_amp, operating_v=-65.0,
                               step_amp=-0.05, step_start=200,
                               step_dur=500):
        duration = step_start + step_dur + 400
        return self.run_current(
            duration=duration,
            dt=self.sim_dt,
            hold_amp=hold_amp,
            wave=None,
            operating_v=operating_v,
            extra_stim={'delay': step_start, 'dur': step_dur, 'amp': step_amp},
            check_spike=True,
        )

    def validate_hh_spiking(self):
        duration = 450
        last = None
        for amp in (0.05, 0.08, 0.10, 0.15, 0.20):
            t, soma, _ = self.run_current(
                duration=duration,
                dt=VALIDATION_DT,
                hold_amp=0.0,
                wave=None,
                operating_v=-65.0,
                extra_stim={'delay': 100, 'dur': 150, 'amp': amp},
                check_spike=False,
            )
            last = (amp, t, soma)
            if has_spike(soma):
                return True, amp, t, soma
        amp, t, soma = last
        return False, amp, t, soma


def run_frequency_sweep(neuron, freqs, hold_amp, ac_amp=0.01,
                        operating_v=-65.0):
    results = [
        neuron.single_frequency_response(
            freq,
            ac_amp=ac_amp,
            hold_amp=hold_amp,
            n_cycles=8,
            discard_cycles=3,
            operating_v=operating_v,
        )
        for freq in freqs
    ]
    return {
        'soma_gain': np.array([r['soma_gain'] for r in results]),
        'dend_gain': np.array([r['dend_gain'] for r in results]),
        'soma_amp': np.array([r['soma_amp'] for r in results]),
        'dend_amp': np.array([r['dend_amp'] for r in results]),
        'ac_amp': ac_amp,
    }


def baseline_is_stable(neuron, hold_amp, operating_v):
    t, soma, dend = neuron.run_current(
        duration=1200,
        hold_amp=hold_amp,
        wave=None,
        operating_v=operating_v,
        check_spike=False,
    )
    steady = float(np.mean(soma[t > 1000]))
    return (not has_spike(soma)) and (not has_spike(dend)) and abs(steady - operating_v) < 1.0
