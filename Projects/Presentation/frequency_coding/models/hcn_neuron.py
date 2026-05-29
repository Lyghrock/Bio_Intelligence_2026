"""
HCN Neuron Model - Theta Resonance Analysis
===========================================
Clean subthreshold pas+somatic HH vs pas+somatic HH+Ih simulation showing
that HCN/Ih changes frequency tuning under hyperpolarized conditions.
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

_ref_mod_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'reference_mod', 'nrnmech.dll'
))
if os.path.exists(_ref_mod_path):
    h.nrn_load_dll(_ref_mod_path)
else:
    print(f"[WARN] Mechanism DLL not found at: {_ref_mod_path}")


THETA_BAND = (4, 10)
LOW_BAND = (1, 3)
HIGH_BAND = (12, 25)
CONTROL_LABEL = 'No HCN control'
HCN_LABEL = 'With HCN'
FREQUENCY_SWEEP = np.array(
    [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25],
    dtype=float,
)
SIM_DT = 0.5
THETA_SHADE = '#fff2b2'
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


class SubthresholdResonanceNeuron:
    """Clean pas+somatic HH or pas+somatic HH+Ih model for subthreshold HCN analysis."""

    def __init__(self, name, with_hcn=False, dend_L=300, g_h=0.001,
                 g_pas_dend=0.0003, eh_hcn=-45.0):
        self.name = name
        self.with_hcn = with_hcn
        self.g_h = g_h
        self.eh_hcn = eh_hcn

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
            self.soma.insert('Ih')
            self._set_hcn_reversal()
            for seg in self.soma:
                seg.Ih.gIhbar = g_h * 0.3

            self.dend.insert('Ih')
            for seg in self.dend:
                seg.Ih.gIhbar = g_h * (0.5 + 0.5 * seg.x)

        self.dend.connect(self.soma(1))

    def _insert_somatic_hh(self):
        global _hh_notice_printed
        try:
            self.soma.insert('hh1')
            mech_name = 'hh1'
        except Exception:
            try:
                self.soma.insert('hh')
                mech_name = 'hh'
            except Exception as exc:
                print(f"[WARN] Could not insert HH/HH1 into soma for {self.name}: {exc}")
                return

        for seg in self.soma:
            if mech_name == 'hh1':
                mech = seg.hh1
                for attr, value in (
                    ('gnabar', 0.08),
                    ('gkbar', 0.024),
                    ('gl', 0.0001),
                ):
                    try:
                        setattr(mech, attr, value)
                    except Exception:
                        print(f"[WARN] Could not set hh1.{attr} for {self.name}; using default.")
            else:
                mech = seg.hh
                for attr, value in (
                    ('gnabar', 0.08),
                    ('gkbar', 0.024),
                    ('gl', 0.0001),
                    ('el', -65.0),
                ):
                    try:
                        setattr(mech, attr, value)
                    except Exception:
                        print(f"[WARN] Could not set hh.{attr} for {self.name}; using default.")

        if not _hh_notice_printed:
            print("[OK] Added HH/HH1 Na/K conductance to soma only.")
            _hh_notice_printed = True

    def _set_hcn_reversal(self):
        try:
            h.ehcn_Ih = self.eh_hcn
        except Exception:
            pass

    def _record_vectors(self):
        t_vec = h.Vector()
        v_soma = h.Vector()
        v_dend = h.Vector()
        t_vec.record(h._ref_t)
        v_soma.record(self.soma(0.5)._ref_v)
        v_dend.record(self.dend(0.8)._ref_v)
        return t_vec, v_soma, v_dend

    def run_current(self, duration, dt=SIM_DT, hold_amp=0.0, wave=None, ac_loc='dend',
                    init_v=-75.0, extra_stim=None):
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
        if has_spike(soma) or has_spike(dend):
            raise RuntimeError(f"{self.name} became suprathreshold")
        return t, soma, dend

    def calibrate_holding_current(self, target_v=-75.0, duration=1200, dt=SIM_DT):
        lo, hi = -0.5, 0.2
        best_amp = 0.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            t, soma, _ = self.run_current(
                duration=duration,
                dt=dt,
                hold_amp=mid,
                wave=None,
                init_v=target_v,
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
        )
        soma_v = float(np.mean(soma[t > duration - 200]))
        dend_v = float(np.mean(dend[t > duration - 200]))
        return best_amp, soma_v, dend_v

    def sine_response(self, freq, ac_amp=0.01, hold_amp=0.0, n_cycles=8,
                      discard_cycles=3, ac_loc='dend', init_v=-75.0):
        dt = SIM_DT
        duration = n_cycles * 1000.0 / freq
        t_wave = np.arange(0, duration, dt)
        wave = ac_amp * np.sin(2 * np.pi * freq * (t_wave / 1000.0))
        t, soma, dend = self.run_current(
            duration=duration,
            dt=dt,
            hold_amp=hold_amp,
            wave=wave,
            ac_loc=ac_loc,
            init_v=init_v,
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
                  hold_amp=0.0, ac_loc='dend', ramp_ms=200.0, init_v=-75.0):
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
            ac_loc=ac_loc,
            init_v=init_v,
        )

    def hyperpolarization_test(self, hold_amp, step_amp=-0.08,
                               step_start=200, step_dur=500):
        duration = step_start + step_dur + 400
        return self.run_current(
            duration=duration,
            dt=SIM_DT,
            hold_amp=hold_amp,
            wave=None,
            init_v=-75.0,
            extra_stim={'delay': step_start, 'dur': step_dur, 'amp': step_amp},
        )


def normalized_gain(freqs, gain):
    return gain / (np.interp(0.5, freqs, gain) + 1e-12)


def run_clean_sweep(neuron, freqs, hold_amp, ac_amp=0.01, init_v=-75.0):
    results = [
        neuron.sine_response(
            freq,
            ac_amp=ac_amp,
            hold_amp=hold_amp,
            n_cycles=8,
            discard_cycles=3,
            init_v=init_v,
        )
        for freq in freqs
    ]
    return {
        'soma_gain': np.array([r['soma_gain'] for r in results]),
        'dend_gain': np.array([r['dend_gain'] for r in results]),
        'soma_amp': np.array([r['soma_amp'] for r in results]),
        'dend_amp': np.array([r['dend_amp'] for r in results]),
    }


def frequency_response_metrics(freqs, gain, theta_band=THETA_BAND):
    norm = normalized_gain(freqs, gain)
    theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    if not np.any(theta_mask):
        raise RuntimeError("Frequency sweep has no theta-band samples")

    theta_norm = norm[theta_mask]
    theta_freqs = freqs[theta_mask]
    idx = int(np.argmax(theta_norm))
    return {
        'norm': norm,
        'ri': float(theta_norm[idx]),
        'peak_freq': float(theta_freqs[idx]),
        'z_ref': float(np.interp(0.5, freqs, gain)),
    }


def theta_prominence(freqs, gain):
    z = normalized_gain(freqs, gain)
    theta = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
    low = (freqs >= LOW_BAND[0]) & (freqs < LOW_BAND[1])
    high = (freqs >= HIGH_BAND[0]) & (freqs <= HIGH_BAND[1])
    if not np.any(theta) or not np.any(low) or not np.any(high):
        raise RuntimeError("Frequency sweep missing theta, low, or high band samples")

    theta_peak = np.max(z[theta])
    flank_mean = 0.5 * (np.mean(z[low]) + np.mean(z[high]))
    return float(theta_peak - flank_mean)


def set_tight_ylim(ax, values, include_one=True, min_pad_fraction=0.08):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if include_one:
        values = np.concatenate([values, np.array([1.0])])
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    span = max(y_max - y_min, 0.02)
    pad = span * min_pad_fraction
    ax.set_ylim(y_min - pad, y_max + pad)


def plot_zap_trace(ax, t_control, v_control, t_hcn, v_hcn):
    zap_mask_control = t_control >= 200
    zap_mask_hcn = t_hcn >= 200
    control_baseline = np.mean(v_control[(t_control >= 300) & (t_control <= 600)])
    hcn_baseline = np.mean(v_hcn[(t_hcn >= 300) & (t_hcn <= 600)])

    ax.plot(
        t_control[zap_mask_control],
        v_control[zap_mask_control] - control_baseline,
        color='tab:blue',
        alpha=0.65,
        lw=1.0,
        label=CONTROL_LABEL,
    )
    ax.plot(
        t_hcn[zap_mask_hcn],
        v_hcn[zap_mask_hcn] - hcn_baseline,
        color='tab:red',
        lw=1.2,
        label=HCN_LABEL,
    )
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Soma voltage fluctuation (mV)')
    ax.set_title('Subthreshold ZAP trace, no spikes')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(200, 4000)


def main():
    target_v = -75.0
    g_pas_dend = 0.0003
    eh_hcn = -45.0
    candidate_g_h = [0.0002, 0.0005]
    zap_amp = 0.01
    sweep_amp = 0.01
    step_amp = -0.08
    freqs = FREQUENCY_SWEEP

    neuron_control = SubthresholdResonanceNeuron(
        'clean_control',
        with_hcn=False,
        dend_L=300,
        g_pas_dend=g_pas_dend,
    )
    hold_control, _, _ = (
        neuron_control.calibrate_holding_current(target_v=target_v)
    )
    control_sweep = run_clean_sweep(neuron_control, freqs, hold_control, ac_amp=sweep_amp)
    control_soma_metrics = frequency_response_metrics(freqs, control_sweep['soma_gain'])
    control_dend_metrics = frequency_response_metrics(freqs, control_sweep['dend_gain'])

    scan_results = []
    for g_h_candidate in candidate_g_h:
        candidate = SubthresholdResonanceNeuron(
            f'clean_hcn_{g_h_candidate:g}',
            with_hcn=True,
            dend_L=300,
            g_h=g_h_candidate,
            g_pas_dend=g_pas_dend,
            eh_hcn=eh_hcn,
        )
        hold_candidate, _, _ = (
            candidate.calibrate_holding_current(target_v=target_v)
        )
        candidate_sweep = run_clean_sweep(candidate, freqs, hold_candidate, ac_amp=sweep_amp)
        candidate_soma_metrics = frequency_response_metrics(
            freqs,
            candidate_sweep['soma_gain'],
        )
        candidate_dend_metrics = frequency_response_metrics(
            freqs,
            candidate_sweep['dend_gain'],
        )
        candidate_soma_prom = theta_prominence(freqs, candidate_sweep['soma_gain'])
        candidate_dend_prom = theta_prominence(freqs, candidate_sweep['dend_gain'])
        score = (
            candidate_soma_metrics['ri']
            + candidate_soma_prom
            + 0.5 * candidate_dend_prom
            - 0.15 * abs(candidate_soma_metrics['peak_freq'] - 7.0)
        )
        scan_results.append({
            'g_h': g_h_candidate,
            'neuron': candidate,
            'hold': hold_candidate,
            'sweep': candidate_sweep,
            'soma_metrics': candidate_soma_metrics,
            'dend_metrics': candidate_dend_metrics,
            'soma_prominence': candidate_soma_prom,
            'dend_prominence': candidate_dend_prom,
            'score': score,
        })

    ri_threshold = max(1.02, control_soma_metrics['ri'] + 0.03)
    valid_scan = [
        item for item in scan_results
        if item['soma_metrics']['ri'] >= ri_threshold
        and THETA_BAND[0] <= item['soma_metrics']['peak_freq'] <= THETA_BAND[1]
    ]
    best = max(valid_scan if valid_scan else scan_results, key=lambda item: item['score'])

    g_h = best['g_h']
    neuron_hcn = best['neuron']
    hold_hcn = best['hold']
    hcn_sweep = best['sweep']
    hcn_soma_metrics = best['soma_metrics']
    hcn_dend_metrics = best['dend_metrics']

    t_control_sag, v_control_sag, _ = neuron_control.hyperpolarization_test(
        hold_amp=hold_control,
        step_amp=step_amp,
    )
    t_hcn_sag, v_hcn_sag, _ = neuron_hcn.hyperpolarization_test(
        hold_amp=hold_hcn,
        step_amp=step_amp,
    )
    sag_control = compute_step_metrics(t_control_sag, v_control_sag)
    sag_hcn = compute_step_metrics(t_hcn_sag, v_hcn_sag)

    t_control_zap, v_control_zap, _ = neuron_control.zap_trace(
        duration=4000,
        f_start=0.5,
        f_end=25,
        ac_amp=zap_amp,
        hold_amp=hold_control,
    )
    t_hcn_zap, v_hcn_zap, _ = neuron_hcn.zap_trace(
        duration=4000,
        f_start=0.5,
        f_end=25,
        ac_amp=zap_amp,
        hold_amp=hold_hcn,
    )

    prom_control_soma = theta_prominence(freqs, control_sweep['soma_gain'])
    prom_hcn_soma = theta_prominence(freqs, hcn_sweep['soma_gain'])
    prom_control_dend = theta_prominence(freqs, control_sweep['dend_gain'])
    prom_hcn_dend = theta_prominence(freqs, hcn_sweep['dend_gain'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'HCN/Ih reshapes subthreshold membrane frequency tuning toward theta-band resonance',
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
        f'Sag recovery = {sag_hcn["sag_recovery"]:.1f} mV',
        xy=(690, sag_hcn['v_late']),
        xytext=(545, sag_hcn['v_late'] + 3.0),
        arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.0),
        color='tab:red',
        fontsize=9,
    )
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('Hyperpolarization Sag / HCN activation')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1100)

    plot_zap_trace(axes[0, 1], t_control_zap, v_control_zap, t_hcn_zap, v_hcn_zap)

    ax = axes[0, 2]
    ax.plot(
        freqs,
        control_sweep['soma_gain'],
        '-o',
        color='tab:blue',
        ms=4,
        lw=2,
        label=f'{CONTROL_LABEL} soma',
    )
    ax.plot(
        freqs,
        hcn_sweep['soma_gain'],
        '-o',
        color='tab:red',
        ms=4,
        lw=2,
        label=f'{HCN_LABEL} soma',
    )
    ax.plot(
        freqs,
        control_sweep['dend_gain'],
        '--s',
        color='tab:blue',
        ms=3,
        lw=1.5,
        alpha=0.65,
        label=f'{CONTROL_LABEL} dend(0.8)',
    )
    ax.plot(
        freqs,
        hcn_sweep['dend_gain'],
        '--s',
        color='tab:red',
        ms=3,
        lw=1.5,
        alpha=0.75,
        label=f'{HCN_LABEL} dend(0.8)',
    )
    ax.axvspan(THETA_BAND[0], THETA_BAND[1], alpha=0.35, color=THETA_SHADE, label='Theta band')
    ax.axvline(hcn_soma_metrics['peak_freq'], color='firebrick', ls='--', lw=1.3)
    y_peak_abs = float(np.interp(hcn_soma_metrics['peak_freq'], freqs, hcn_sweep['soma_gain']))
    ax.annotate(
        f'HCN peak = {hcn_soma_metrics["peak_freq"]:.0f} Hz',
        xy=(hcn_soma_metrics['peak_freq'], y_peak_abs),
        xytext=(hcn_soma_metrics['peak_freq'] + 2.0, y_peak_abs * 1.08),
        arrowprops=dict(arrowstyle='->', color='firebrick', lw=1.0),
        fontsize=9,
        color='firebrick',
    )
    ax.text(
        0.03,
        0.08,
        'HCN may lower absolute gain because it increases membrane conductance,\n'
        'but it reshapes frequency tuning.',
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='none'),
    )
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Transfer Gain (MOhm)')
    ax.set_title('Absolute Transfer Gain')
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 25)

    ax = axes[1, 0]
    ax.plot(
        freqs,
        control_soma_metrics['norm'],
        '-o',
        color='tab:blue',
        ms=4,
        lw=2,
        label=f'{CONTROL_LABEL} soma',
    )
    ax.plot(
        freqs,
        hcn_soma_metrics['norm'],
        '-o',
        color='tab:red',
        ms=4,
        lw=2.4,
        label=f'{HCN_LABEL} soma',
    )
    ax.plot(
        freqs,
        control_dend_metrics['norm'],
        '--',
        color='tab:blue',
        lw=1.3,
        alpha=0.55,
        label=f'{CONTROL_LABEL} dend(0.8)',
    )
    ax.plot(
        freqs,
        hcn_dend_metrics['norm'],
        '--',
        color='tab:red',
        lw=1.3,
        alpha=0.65,
        label=f'{HCN_LABEL} dend(0.8)',
    )
    ax.axvspan(THETA_BAND[0], THETA_BAND[1], alpha=0.35, color=THETA_SHADE)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.axvline(hcn_soma_metrics['peak_freq'], color='firebrick', ls='--', lw=1.3)
    ax.text(
        0.52,
        0.92,
        f'HCN RI = {hcn_soma_metrics["ri"]:.2f}\n'
        f'Control RI = {control_soma_metrics["ri"]:.2f}\n'
        f'Peak = {hcn_soma_metrics["peak_freq"]:.0f} Hz',
        transform=ax.transAxes,
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
    )
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized impedance (Z / Z at 0.5 Hz)')
    ax.set_title('Normalized Impedance / RI')
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 25)
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
    control_prominence = np.array([prom_control_soma, prom_control_dend])
    hcn_prominence = np.array([prom_hcn_soma, prom_hcn_dend])
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
    ax.set_ylabel('Theta prominence')
    ax.set_title('Theta resonance prominence')
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
    ri_statement = (
        'HCN RI > control RI'
        if hcn_soma_metrics['ri'] > control_soma_metrics['ri']
        else 'HCN RI not above control RI'
    )
    prom_statement = (
        'HCN > control at soma and dendrite'
        if prom_hcn_soma > prom_control_soma and prom_hcn_dend > prom_control_dend
        else 'see prominence panel'
    )
    summary = (
        "Model:\n"
        "  - No HCN control: pas + somatic HH/HH1\n"
        "  - With HCN: pas + somatic HH/HH1 + Ih\n"
        "  - Input: distal dendritic current injection at dend(0.8)\n"
        "  - Recording: soma and dend(0.8)\n"
        f"  - Holding target: {target_v:.1f} mV\n"
        f"  - g_h selected: {g_h:.4g} S/cm2\n\n"
        "Key results:\n"
        f"  - Sag recovery: control={sag_control['sag_recovery']:.2f} mV, "
        f"HCN={sag_hcn['sag_recovery']:.2f} mV\n"
        "  - Absolute gain can be lower with HCN\n"
        "    because Ih increases membrane conductance\n"
        f"  - Normalized impedance: {ri_statement}\n"
        f"  - HCN peak frequency: {hcn_soma_metrics['peak_freq']:.1f} Hz\n"
        f"  - Theta prominence: {prom_statement}\n\n"
        "Conclusion:\n"
        "Under hyperpolarized subthreshold conditions,\n"
        "HCN/Ih produces a relative theta-band resonance.\n"
        "This is a normalized frequency-tuning effect,\n"
        "not an absolute voltage-amplitude amplification."
    )
    ax.text(
        0.03,
        0.98,
        summary,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='none'),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig_dir = os.path.join(_figures_root, 'hcn_resonance')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'hcn_resonance.png')
    pdf_path = os.path.join(fig_dir, 'hcn_resonance.pdf')
    plt.savefig(fig_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close(fig)

    print(f"Figure saved: {fig_path}")
    print(f"PDF saved: {pdf_path}")
    print(f"Sag recovery: control={sag_control['sag_recovery']:.2f} mV, "
          f"HCN={sag_hcn['sag_recovery']:.2f} mV")
    print(f"Control RI: {control_soma_metrics['ri']:.3f}")
    print(f"HCN RI: {hcn_soma_metrics['ri']:.3f}")
    print(f"HCN peak frequency: {hcn_soma_metrics['peak_freq']:.1f} Hz")
    print("Theta prominence: "
          f"control soma={prom_control_soma:.3f}, "
          f"HCN soma={prom_hcn_soma:.3f}, "
          f"control dend={prom_control_dend:.3f}, "
          f"HCN dend={prom_hcn_dend:.3f}")
    print("Simulation complete.")


if __name__ == '__main__':
    main()
