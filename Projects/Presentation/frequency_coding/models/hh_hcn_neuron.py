"""
HH + HCN Neuron Model - Near-Threshold Subthreshold Theta Resonance
===================================================================
Independent active-membrane resonance analysis. The resonance sweep is kept
subthreshold; a separate suprathreshold pulse validates HH excitability.
"""

from neuron import h
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_figures_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))

h.load_file('stdrun.hoc')
h.CVode().active(0)
h.celsius = 6.3

_ref_mod_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'reference_mod', 'nrnmech.dll'
))
if os.path.exists(_ref_mod_path):
    h.nrn_load_dll(_ref_mod_path)
else:
    print(f"[WARN] Mechanism DLL not found at: {_ref_mod_path}")


THETA_BAND = (4, 10)
FREQUENCY_SWEEP = np.array(
    [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12,
     15, 20, 25, 30, 35, 40, 50, 60, 80, 100],
    dtype=float,
)
SELECTED_TARGET_V = -65.0
SELECTED_G_H = 0.0008
SELECTED_AC_AMP = 0.01
SIM_DT = 0.2
VALIDATION_DT = 0.025
THETA_SHADE = '#fff2b2'
CONTROL_LABEL = 'HH control'
HCN_LABEL = 'HH + HCN'


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


def frequency_response_metrics(freqs, gain):
    z_norm = normalized_gain(freqs, gain)
    peak_idx = int(np.argmax(z_norm))
    theta = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
    if not np.any(theta):
        raise RuntimeError("Frequency sweep has no theta-band samples")
    return {
        'norm': z_norm,
        'ri': float(z_norm[peak_idx]),
        'peak_freq': float(freqs[peak_idx]),
        'theta_ri': float(np.max(z_norm[theta])),
        'z_ref': float(np.interp(0.5, freqs, gain)),
    }


def active_resonance_band(freqs, peak_freq):
    band_low = max(4.0, peak_freq - 10.0)
    band_high = min(float(freqs[-1]), peak_freq + 10.0)
    if band_high - band_low > 24.0:
        center = 0.5 * (band_low + band_high)
        band_low = max(4.0, center - 12.0)
        band_high = min(float(freqs[-1]), center + 12.0)
    return float(band_low), float(band_high)


def resonance_prominence(freqs, gain, active_band):
    z = normalized_gain(freqs, gain)
    active = (freqs >= active_band[0]) & (freqs <= active_band[1])
    low = (freqs >= 1) & (freqs < active_band[0])
    high = (freqs > active_band[1]) & (freqs <= freqs[-1])
    if not np.any(active) or not np.any(low):
        raise RuntimeError("Frequency sweep missing active or low-frequency flank samples")
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


class HHHCNResonanceNeuron:
    """Soma HH active membrane with optional HCN/Ih in soma and dendrite."""

    def __init__(self, name, with_hcn=False, g_h=0.0002, eh_hcn=-45.0,
                 g_pas_dend=0.0003):
        self.name = name
        self.with_hcn = with_hcn
        self.g_h = g_h
        self.eh_hcn = eh_hcn
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

        self._insert_hh()

        self.dend.L = 300
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
                seg.Ih.gIhbar = g_h * 0.25
            self.dend.insert('Ih')
            for seg in self.dend:
                seg.Ih.gIhbar = g_h * (0.5 + 0.5 * seg.x)

        self.dend.connect(self.soma(1))

    def _insert_hh(self):
        try:
            self.soma.insert('hh1')
            self.hh_mech = 'hh1'
        except Exception:
            self.soma.insert('hh')
            self.hh_mech = 'hh'
            print(f"[WARN] {self.name}: hh1 unavailable; using built-in hh")

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
                    print(f"[WARN] {self.name}: could not set {self.hh_mech}.{attr}")

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

    def run_current(self, duration, dt=SIM_DT, hold_amp=0.0, wave=None,
                    ac_loc='dend', init_v=-65.0, extra_stim=None,
                    check_spike=True):
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
        h.finitialize(init_v)
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

    def calibrate_holding_current(self, target_v=-62.0, duration=1200, dt=SIM_DT):
        lo, hi = -0.6, 0.4
        best_amp = 0.0
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            t, soma, _ = self.run_current(
                duration=duration,
                dt=dt,
                hold_amp=mid,
                wave=None,
                init_v=target_v,
                check_spike=False,
            )
            steady = float(np.mean(soma[t > duration - 200]))
            best_amp = mid
            if steady > target_v:
                hi = mid
            else:
                lo = mid

        t, soma, dend = self.run_current(
            duration=duration,
            dt=dt,
            hold_amp=best_amp,
            wave=None,
            init_v=target_v,
            check_spike=False,
        )
        soma_v = float(np.mean(soma[t > duration - 200]))
        dend_v = float(np.mean(dend[t > duration - 200]))
        return best_amp, soma_v, dend_v

    def zap_trace(self, duration=4000, f_start=0.5, f_end=25, ac_amp=0.005,
                  hold_amp=0.0, ramp_ms=200.0, init_v=-62.0):
        dt = SIM_DT
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
            ac_loc='dend',
            init_v=init_v,
            check_spike=True,
        )

    def single_frequency_response(self, freq, ac_amp=0.005, hold_amp=0.0,
                                  n_cycles=8, discard_cycles=3, init_v=-62.0):
        dt = SIM_DT
        duration = n_cycles * 1000.0 / freq
        t_wave = np.arange(0, duration, dt)
        wave = ac_amp * np.sin(2 * np.pi * freq * (t_wave / 1000.0))
        t, soma, dend = self.run_current(
            duration=duration,
            dt=dt,
            hold_amp=hold_amp,
            wave=wave,
            ac_loc='dend',
            init_v=init_v,
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

    def hyperpolarization_test(self, hold_amp, target_v=-62.0, step_amp=-0.05,
                               step_start=200, step_dur=500):
        duration = step_start + step_dur + 400
        return self.run_current(
            duration=duration,
            dt=SIM_DT,
            hold_amp=hold_amp,
            wave=None,
            init_v=target_v,
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
                init_v=-65.0,
                extra_stim={'delay': 100, 'dur': 150, 'amp': amp},
                check_spike=False,
            )
            last = (amp, t, soma)
            if has_spike(soma):
                return True, amp, t, soma
        amp, t, soma = last
        return False, amp, t, soma


def calibrate_holding_current(neuron, target_v=-62.0):
    return neuron.calibrate_holding_current(target_v=target_v)


def run_current(neuron, *args, **kwargs):
    return neuron.run_current(*args, **kwargs)


def zap_trace(neuron, *args, **kwargs):
    return neuron.zap_trace(*args, **kwargs)


def single_frequency_response(neuron, *args, **kwargs):
    return neuron.single_frequency_response(*args, **kwargs)


def validate_hh_spiking(neuron):
    return neuron.validate_hh_spiking()


def run_frequency_sweep(neuron, freqs, hold_amp, ac_amp=0.005, target_v=-62.0):
    results = [
        neuron.single_frequency_response(
            freq,
            ac_amp=ac_amp,
            hold_amp=hold_amp,
            n_cycles=8,
            discard_cycles=3,
            init_v=target_v,
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


def baseline_is_stable(neuron, hold_amp, target_v):
    t, soma, dend = neuron.run_current(
        duration=1200,
        dt=SIM_DT,
        hold_amp=hold_amp,
        wave=None,
        init_v=target_v,
        check_spike=False,
    )
    steady = float(np.mean(soma[t > 1000]))
    return (not has_spike(soma)) and (not has_spike(dend)) and abs(steady - target_v) < 1.0


def run_fixed_sweep(neuron, freqs, hold_amp, target_v, ac_amp):
    sweep = run_frequency_sweep(neuron, freqs, hold_amp, ac_amp=ac_amp, target_v=target_v)
    return sweep, ac_amp, True


def run_fixed_zap(neuron, hold_amp, target_v, ac_amp):
    t, soma, dend = neuron.zap_trace(ac_amp=ac_amp, hold_amp=hold_amp, init_v=target_v)
    return t, soma, dend, ac_amp, True


def build_fixed_condition(target_v=SELECTED_TARGET_V, g_h=SELECTED_G_H,
                          ac_amp=SELECTED_AC_AMP, g_pas_dend=0.0003):
    control = HHHCNResonanceNeuron(
        'hh_control_fixed',
        with_hcn=False,
        g_pas_dend=g_pas_dend,
    )
    hold_control, _, _ = calibrate_holding_current(control, target_v=target_v)
    if not baseline_is_stable(control, hold_control, target_v):
        raise RuntimeError(f"HH control baseline unstable at {target_v:.1f} mV")

    control_sweep, control_amp, control_ok = run_fixed_sweep(
        control,
        FREQUENCY_SWEEP,
        hold_control,
        target_v,
        ac_amp,
    )
    control_zap = run_fixed_zap(control, hold_control, target_v, ac_amp)
    control_soma_metrics = frequency_response_metrics(FREQUENCY_SWEEP, control_sweep['soma_gain'])
    control_dend_metrics = frequency_response_metrics(FREQUENCY_SWEEP, control_sweep['dend_gain'])

    hcn = HHHCNResonanceNeuron(
        'hh_hcn_fixed',
        with_hcn=True,
        g_h=g_h,
        g_pas_dend=g_pas_dend,
    )
    hold_hcn, _, _ = calibrate_holding_current(hcn, target_v=target_v)
    if not baseline_is_stable(hcn, hold_hcn, target_v):
        raise RuntimeError(f"HH+HCN baseline unstable at {target_v:.1f} mV")

    hcn_sweep, hcn_amp, hcn_ok = run_fixed_sweep(
        hcn,
        FREQUENCY_SWEEP,
        hold_hcn,
        target_v,
        ac_amp,
    )
    hcn_zap = run_fixed_zap(hcn, hold_hcn, target_v, ac_amp)
    hcn_soma_metrics = frequency_response_metrics(FREQUENCY_SWEEP, hcn_sweep['soma_gain'])
    hcn_dend_metrics = frequency_response_metrics(FREQUENCY_SWEEP, hcn_sweep['dend_gain'])
    active_band = active_resonance_band(FREQUENCY_SWEEP, hcn_soma_metrics['peak_freq'])
    control_soma_prom = resonance_prominence(FREQUENCY_SWEEP, control_sweep['soma_gain'], active_band)
    control_dend_prom = resonance_prominence(FREQUENCY_SWEEP, control_sweep['dend_gain'], active_band)
    hcn_soma_prom = resonance_prominence(FREQUENCY_SWEEP, hcn_sweep['soma_gain'], active_band)
    hcn_dend_prom = resonance_prominence(FREQUENCY_SWEEP, hcn_sweep['dend_gain'], active_band)

    return {
        'target_v': target_v,
        'control': control,
        'hcn': hcn,
        'g_h': g_h,
        'hold_control': hold_control,
        'hold_hcn': hold_hcn,
        'control_sweep': control_sweep,
        'hcn_sweep': hcn_sweep,
        'control_sweep_amp': control_amp,
        'hcn_sweep_amp': hcn_amp,
        'control_zap': control_zap,
        'hcn_zap': hcn_zap,
        'control_soma_metrics': control_soma_metrics,
        'control_dend_metrics': control_dend_metrics,
        'hcn_soma_metrics': hcn_soma_metrics,
        'hcn_dend_metrics': hcn_dend_metrics,
        'active_band': active_band,
        'control_soma_prom': control_soma_prom,
        'control_dend_prom': control_dend_prom,
        'hcn_soma_prom': hcn_soma_prom,
        'hcn_dend_prom': hcn_dend_prom,
        'control_ok': control_ok,
        'hcn_ok': hcn_ok,
    }


def plot_zap_trace(ax, t_control, v_control, t_hcn, v_hcn):
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
        label=CONTROL_LABEL,
    )
    ax.plot(
        t_hcn[mask_hcn],
        v_hcn[mask_hcn] - baseline_hcn,
        color='tab:red',
        lw=1.2,
        label=HCN_LABEL,
    )
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Soma voltage fluctuation (mV)')
    ax.set_title('B  Subthreshold ZAP trace, no spikes')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(200, 4000)


def main():
    best = build_fixed_condition()
    target_v = best['target_v']
    control = best['control']
    hcn = best['hcn']
    g_h = best['g_h']
    hold_control = best['hold_control']
    hold_hcn = best['hold_hcn']
    control_sweep = best['control_sweep']
    hcn_sweep = best['hcn_sweep']
    control_sweep_amp = best['control_sweep_amp']
    hcn_sweep_amp = best['hcn_sweep_amp']
    active_band = best['active_band']
    control_soma_metrics = best['control_soma_metrics']
    control_dend_metrics = best['control_dend_metrics']
    hcn_soma_metrics = best['hcn_soma_metrics']
    hcn_dend_metrics = best['hcn_dend_metrics']
    control_soma_prom = best['control_soma_prom']
    control_dend_prom = best['control_dend_prom']
    hcn_soma_prom = best['hcn_soma_prom']
    hcn_dend_prom = best['hcn_dend_prom']

    hh_validation_spike, validation_amp, _, _ = validate_hh_spiking(control)

    t_control_sag, v_control_sag, _ = control.hyperpolarization_test(
        hold_amp=hold_control,
        target_v=target_v,
        step_amp=-0.05,
    )
    t_hcn_sag, v_hcn_sag, _ = hcn.hyperpolarization_test(
        hold_amp=hold_hcn,
        target_v=target_v,
        step_amp=-0.05,
    )
    sag_control = compute_step_metrics(t_control_sag, v_control_sag)
    sag_hcn = compute_step_metrics(t_hcn_sag, v_hcn_sag)

    t_control_zap, v_control_zap, _, control_zap_amp, control_zap_ok = best['control_zap']
    t_hcn_zap, v_hcn_zap, _, hcn_zap_amp, hcn_zap_ok = best['hcn_zap']
    zap_spike_free = control_zap_ok and hcn_zap_ok
    sweep_spike_free = best['control_ok'] and best['hcn_ok']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'HH + HCN/Ih model: near-threshold subthreshold active resonance',
        fontsize=14,
        y=0.98,
    )

    ax = axes[0, 0]
    ax.plot(
        t_control_sag,
        v_control_sag,
        color='tab:blue',
        alpha=0.75,
        lw=1.5,
        label=f'{CONTROL_LABEL}, V_rest={sag_control["v_rest"]:.1f} mV',
    )
    ax.plot(
        t_hcn_sag,
        v_hcn_sag,
        color='tab:red',
        lw=2,
        label=f'{HCN_LABEL}, V_rest={sag_hcn["v_rest"]:.1f} mV',
    )
    ax.axvspan(200, 700, alpha=0.08, color='gray')
    ax.annotate(
        f'HCN sag recovery = {sag_hcn["sag_recovery"]:.1f} mV',
        xy=(690, sag_hcn['v_late']),
        xytext=(505, sag_hcn['v_late'] + 2.0),
        arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.0),
        color='tab:red',
        fontsize=9,
    )
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('A  Hyperpolarization sag with HH background')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1100)

    plot_zap_trace(axes[0, 1], t_control_zap, v_control_zap, t_hcn_zap, v_hcn_zap)

    ax = axes[0, 2]
    ax.plot(
        FREQUENCY_SWEEP,
        control_sweep['soma_gain'],
        '-o',
        color='tab:blue',
        ms=4,
        lw=2,
        label=f'{CONTROL_LABEL} soma',
    )
    ax.plot(
        FREQUENCY_SWEEP,
        hcn_sweep['soma_gain'],
        '-o',
        color='tab:red',
        ms=4,
        lw=2,
        label=f'{HCN_LABEL} soma',
    )
    ax.plot(
        FREQUENCY_SWEEP,
        control_sweep['dend_gain'],
        '--s',
        color='tab:blue',
        ms=3,
        lw=1.5,
        alpha=0.65,
        label=f'{CONTROL_LABEL} dend(0.8)',
    )
    ax.plot(
        FREQUENCY_SWEEP,
        hcn_sweep['dend_gain'],
        '--s',
        color='tab:red',
        ms=3,
        lw=1.5,
        alpha=0.75,
        label=f'{HCN_LABEL} dend(0.8)',
    )
    ax.axvspan(
        active_band[0],
        active_band[1],
        alpha=0.35,
        color=THETA_SHADE,
        label='HH-shifted resonance band',
    )
    ax.axvline(hcn_soma_metrics['peak_freq'], color='firebrick', ls='--', lw=1.3)
    ax.text(
        0.03,
        0.08,
        'HH background shifts the enhanced response to a higher frequency band;\n'
        'absolute gain is not the resonance criterion.',
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='none'),
    )
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Transfer gain (MOhm)')
    ax.set_title('C  Absolute transfer gain')
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, FREQUENCY_SWEEP[-1])

    ax = axes[1, 0]
    ax.plot(
        FREQUENCY_SWEEP,
        control_soma_metrics['norm'],
        '-o',
        color='tab:blue',
        ms=4,
        lw=2,
        label=f'{CONTROL_LABEL} soma',
    )
    ax.plot(
        FREQUENCY_SWEEP,
        hcn_soma_metrics['norm'],
        '-o',
        color='tab:red',
        ms=4,
        lw=2.4,
        label=f'{HCN_LABEL} soma',
    )
    ax.plot(
        FREQUENCY_SWEEP,
        control_dend_metrics['norm'],
        '--',
        color='tab:blue',
        lw=1.3,
        alpha=0.55,
        label=f'{CONTROL_LABEL} dend(0.8)',
    )
    ax.plot(
        FREQUENCY_SWEEP,
        hcn_dend_metrics['norm'],
        '--',
        color='tab:red',
        lw=1.3,
        alpha=0.65,
        label=f'{HCN_LABEL} dend(0.8)',
    )
    ax.axvspan(active_band[0], active_band[1], alpha=0.35, color=THETA_SHADE)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.axvline(hcn_soma_metrics['peak_freq'], color='firebrick', ls='--', lw=1.3)
    ax.text(
        0.52,
        0.92,
        f'HH control global RI = {control_soma_metrics["ri"]:.2f}\n'
        f'HH+HCN global RI = {hcn_soma_metrics["ri"]:.2f}\n'
        f'HCN peak = {hcn_soma_metrics["peak_freq"]:.0f} Hz',
        transform=ax.transAxes,
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
    )
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized impedance (Z / Z at 0.5 Hz)')
    ax.set_title('D  Normalized impedance / active resonance index')
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, FREQUENCY_SWEEP[-1])
    set_tight_ylim(
        ax,
        np.r_[
            control_soma_metrics['norm'],
            hcn_soma_metrics['norm'],
            control_dend_metrics['norm'],
            hcn_dend_metrics['norm'],
        ],
    )

    ax = axes[1, 1]
    x = np.arange(2)
    width = 0.36
    control_prominence = np.array([control_soma_prom, control_dend_prom])
    hcn_prominence = np.array([hcn_soma_prom, hcn_dend_prom])
    bars_control = ax.bar(
        x - width / 2,
        control_prominence,
        width,
        color='tab:blue',
        label=CONTROL_LABEL,
    )
    bars_hcn = ax.bar(
        x + width / 2,
        hcn_prominence,
        width,
        color='tab:red',
        label=HCN_LABEL,
    )
    ax.axhline(0.0, color='k', ls='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Soma', 'Dendrite (0.8)'])
    ax.set_ylabel('Resonance prominence')
    ax.set_title('E  Resonance prominence')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, axis='y', alpha=0.3)
    ax.bar_label(bars_control, fmt='%.3f', fontsize=8, padding=2)
    ax.bar_label(bars_hcn, fmt='%.3f', fontsize=8, padding=2)
    prom_min = min(np.min(control_prominence), np.min(hcn_prominence), 0.0)
    prom_max = max(np.max(control_prominence), np.max(hcn_prominence), 0.0)
    prom_span = max(prom_max - prom_min, 0.02)
    ax.set_ylim(prom_min - 0.15 * prom_span, prom_max + 0.22 * prom_span)

    ax = axes[1, 2]
    ax.axis('off')
    if hcn_soma_metrics['ri'] > control_soma_metrics['ri']:
        ri_conclusion = 'HH+HCN RI > HH control RI'
    else:
        ri_conclusion = 'HH+HCN RI not above HH control RI'
    if target_v <= -68.0:
        hh_note = 'HH effects are expected to be weak at this hyperpolarized Vm.'
    else:
        hh_note = 'Near-threshold no-spike operating point selected.'
    summary = (
        "Model:\n"
        f"  - Soma: pas + {control.hh_mech} Na/K +/- Ih\n"
        "  - Dendrite: pas +/- Ih\n"
        "  - Input: distal current at dend(0.8)\n"
        "  - Recording: soma and dend(0.8)\n\n"
        "Operating point:\n"
        f"  - Target Vm: {target_v:.1f} mV\n"
        "  - Holding: current clamp, not voltage clamp\n"
        f"  - AC amplitude: control={control_sweep_amp:.4f} nA, HCN={hcn_sweep_amp:.4f} nA\n"
        f"  - Selected frequency range: {FREQUENCY_SWEEP[0]:.1f}-{FREQUENCY_SWEEP[-1]:.0f} Hz\n"
        f"  - Active band: {active_band[0]:.1f}-{active_band[1]:.1f} Hz\n"
        f"  - {hh_note}\n\n"
        "Validation:\n"
        f"  - HH spike validation: {hh_validation_spike}, current={validation_amp:.2f} nA\n"
        "  - ZAP/sweep for resonance: no spikes\n\n"
        "Results:\n"
        f"  - HCN sag recovery: {sag_hcn['sag_recovery']:.2f} mV\n"
        f"  - HH control global RI: {control_soma_metrics['ri']:.3f}\n"
        f"  - HH+HCN global RI: {hcn_soma_metrics['ri']:.3f}\n"
        f"  - HH+HCN global peak: {hcn_soma_metrics['peak_freq']:.1f} Hz\n"
        f"  - Theta-band RI: {hcn_soma_metrics['theta_ri']:.3f}\n"
        f"  - Active prominence soma: control={control_soma_prom:.3f}, HCN={hcn_soma_prom:.3f}\n"
        f"  - Active prominence dend: control={control_dend_prom:.3f}, HCN={hcn_dend_prom:.3f}\n\n"
        "Conclusion:\n"
        "HH background shifted the relative resonance peak to a higher active band.\n"
        "The highlighted band marks the detected HH-shifted resonance range.\n"
        f"{ri_conclusion} under near-threshold subthreshold conditions.\n"
        "This reflects normalized frequency tuning,\n"
        "not absolute voltage amplification."
    )
    ax.text(
        0.03,
        0.98,
        summary,
        transform=ax.transAxes,
        fontsize=8.0,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='none'),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig_dir = os.path.join(_figures_root, 'hcn_resonance')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'hh_hcn_resonance.png')
    pdf_path = os.path.join(fig_dir, 'hh_hcn_resonance.pdf')
    plt.savefig(fig_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close(fig)

    print(f"Figure saved: {fig_path}")
    print(f"PDF saved: {pdf_path}")
    print(f"HH mechanism used: {control.hh_mech}")
    print(f"Selected target Vm: {target_v:.1f} mV")
    print(f"Selected g_h: {g_h:.4g} S/cm2")
    print(f"Selected frequency range: {FREQUENCY_SWEEP[0]:.1f}-{FREQUENCY_SWEEP[-1]:.0f} Hz")
    print(f"Active resonance band: {active_band[0]:.1f}-{active_band[1]:.1f} Hz")
    print(f"HH validation spike: {hh_validation_spike}, current={validation_amp:.2f} nA")
    print(f"ZAP spike-free: {zap_spike_free}")
    print(f"Sweep spike-free: {sweep_spike_free}")
    print(f"HH control global RI: {control_soma_metrics['ri']:.3f}")
    print(f"HH+HCN global RI: {hcn_soma_metrics['ri']:.3f}")
    print(f"HH+HCN global peak frequency: {hcn_soma_metrics['peak_freq']:.1f} Hz")
    print(f"HH+HCN theta-band RI: {hcn_soma_metrics['theta_ri']:.3f}")
    print("Active-band prominence: "
          f"control soma={control_soma_prom:.3f}, "
          f"HCN soma={hcn_soma_prom:.3f}, "
          f"control dend={control_dend_prom:.3f}, "
          f"HCN dend={hcn_dend_prom:.3f}")
    print("Simulation complete.")


if __name__ == '__main__':
    main()
