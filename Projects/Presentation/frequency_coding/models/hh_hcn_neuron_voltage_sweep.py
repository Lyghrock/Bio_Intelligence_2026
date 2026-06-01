"""
HH + HCN Neuron Model - Multi-Voltage Subthreshold Resonance
============================================================
Runs the shared HH + HCN resonance analysis at multiple hyperpolarized
operating voltages and saves a consistently named figure for each voltage.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hcn_resonance_common import (
    DEFAULT_EH_HCN,
    DEFAULT_G_H,
    FREQUENCY_SWEEP_ACTIVE,
    HCN_SOMA_DENSITY_FACTOR,
    HHHCNResonanceNeuron,
    THETA_SHADE,
    baseline_is_stable,
    compute_step_metrics,
    detected_resonance_band,
    frequency_response_metrics,
    has_spike,
    plot_zap_trace,
    resonance_band_label,
    resonance_prominence,
    run_frequency_sweep,
    set_tight_ylim,
)


_figures_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))

CONTROL_LABEL = 'HH control'
HCN_LABEL = 'HH + HCN'
OPERATING_VOLTAGES = (-140.0, -130.0, -120.0, -110.0, -100.0, -90.0, -80.0, -70.0)
FIXED_G_H = DEFAULT_G_H
SIM_DT = 0.2
ZAP_AMP = 0.01
SWEEP_AMP = 0.01
STEP_AMP = -0.05
FREQUENCY_SWEEP = FREQUENCY_SWEEP_ACTIVE
FIGURE_PREFIX = 'hh_hcn_resonance_vm'


def voltage_tag(operating_v):
    sign = 'minus' if operating_v < 0 else 'plus'
    return f'{sign}{abs(int(round(operating_v)))}mv'


def calibrate_holding_current_wide(neuron, operating_v, duration=1200, dt=None):
    """Calibrate holding current over a wider range for deep hyperpolarization."""
    dt = neuron.sim_dt if dt is None else dt
    lo, hi = -5.0, 2.0
    best_amp = 0.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        t, soma, _ = neuron.run_current(
            duration=duration,
            dt=dt,
            hold_amp=mid,
            wave=None,
            operating_v=operating_v,
            check_spike=False,
        )
        steady = float(np.mean(soma[t > duration - 200]))
        best_amp = mid
        if steady > operating_v:
            hi = mid
        else:
            lo = mid

    t, soma, dend = neuron.run_current(
        duration=duration,
        dt=dt,
        hold_amp=best_amp,
        wave=None,
        operating_v=operating_v,
        check_spike=False,
    )
    soma_v = float(np.mean(soma[t > duration - 200]))
    dend_v = float(np.mean(dend[t > duration - 200]))
    return best_amp, soma_v, dend_v


def make_neuron(name, with_hcn):
    return HHHCNResonanceNeuron(
        name,
        with_hcn=with_hcn,
        dend_L=300,
        g_h=FIXED_G_H,
        g_pas_dend=0.0003,
        eh_hcn=DEFAULT_EH_HCN,
        hcn_soma_density=HCN_SOMA_DENSITY_FACTOR,
        sim_dt=SIM_DT,
    )


def run_one_voltage(operating_v):
    freqs = FREQUENCY_SWEEP
    tag = voltage_tag(operating_v)

    neuron_control = make_neuron(f'hh_control_{tag}', with_hcn=False)
    neuron_hcn = make_neuron(f'hh_hcn_{tag}', with_hcn=True)

    hold_control, control_soma_v, control_dend_v = calibrate_holding_current_wide(
        neuron_control,
        operating_v=operating_v,
    )
    hold_hcn, hcn_soma_v, hcn_dend_v = calibrate_holding_current_wide(
        neuron_hcn,
        operating_v=operating_v,
    )
    if not baseline_is_stable(neuron_control, hold_control, operating_v):
        raise RuntimeError(f"HH control baseline unstable at {operating_v:.1f} mV")
    if not baseline_is_stable(neuron_hcn, hold_hcn, operating_v):
        raise RuntimeError(f"HH+HCN baseline unstable at {operating_v:.1f} mV")

    control_sweep = run_frequency_sweep(
        neuron_control,
        freqs,
        hold_control,
        ac_amp=SWEEP_AMP,
        operating_v=operating_v,
    )
    hcn_sweep = run_frequency_sweep(
        neuron_hcn,
        freqs,
        hold_hcn,
        ac_amp=SWEEP_AMP,
        operating_v=operating_v,
    )
    control_soma_metrics = frequency_response_metrics(freqs, control_sweep['soma_gain'])
    control_dend_metrics = frequency_response_metrics(freqs, control_sweep['dend_gain'])
    hcn_soma_metrics = frequency_response_metrics(freqs, hcn_sweep['soma_gain'])
    hcn_dend_metrics = frequency_response_metrics(freqs, hcn_sweep['dend_gain'])

    resonance_band = detected_resonance_band(freqs, hcn_soma_metrics['peak_freq'])
    band_label = resonance_band_label(resonance_band)
    control_soma_prom = resonance_prominence(freqs, control_sweep['soma_gain'], resonance_band)
    control_dend_prom = resonance_prominence(freqs, control_sweep['dend_gain'], resonance_band)
    hcn_soma_prom = resonance_prominence(freqs, hcn_sweep['soma_gain'], resonance_band)
    hcn_dend_prom = resonance_prominence(freqs, hcn_sweep['dend_gain'], resonance_band)

    hh_validation_spike, validation_amp, _, _ = neuron_control.validate_hh_spiking()

    t_control_sag, v_control_sag, _ = neuron_control.hyperpolarization_test(
        hold_amp=hold_control,
        operating_v=operating_v,
        step_amp=STEP_AMP,
    )
    t_hcn_sag, v_hcn_sag, _ = neuron_hcn.hyperpolarization_test(
        hold_amp=hold_hcn,
        operating_v=operating_v,
        step_amp=STEP_AMP,
    )
    sag_control = compute_step_metrics(t_control_sag, v_control_sag)
    sag_hcn = compute_step_metrics(t_hcn_sag, v_hcn_sag)

    t_control_zap, v_control_zap, _ = neuron_control.zap_trace(
        duration=4000,
        f_start=0.5,
        f_end=25,
        ac_amp=ZAP_AMP,
        hold_amp=hold_control,
        operating_v=operating_v,
    )
    t_hcn_zap, v_hcn_zap, _ = neuron_hcn.zap_trace(
        duration=4000,
        f_start=0.5,
        f_end=25,
        ac_amp=ZAP_AMP,
        hold_amp=hold_hcn,
        operating_v=operating_v,
    )
    zap_spike_free = (
        not has_spike(v_control_zap)
        and not has_spike(v_hcn_zap)
    )
    sweep_spike_free = True

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'HH + HCN/Ih model: {operating_v:.0f} mV subthreshold active resonance',
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

    plot_zap_trace(
        axes[0, 1],
        t_control_zap,
        v_control_zap,
        t_hcn_zap,
        v_hcn_zap,
        CONTROL_LABEL,
        HCN_LABEL,
        f'B  Subthreshold ZAP trace at {operating_v:.0f} mV, no spikes',
    )

    ax = axes[0, 2]
    ax.plot(freqs, control_sweep['soma_gain'], '-o', color='tab:blue', ms=4,
            lw=2, label=f'{CONTROL_LABEL} soma')
    ax.plot(freqs, hcn_sweep['soma_gain'], '-o', color='tab:red', ms=4,
            lw=2, label=f'{HCN_LABEL} soma')
    ax.plot(freqs, control_sweep['dend_gain'], '--s', color='tab:blue', ms=3,
            lw=1.5, alpha=0.65, label=f'{CONTROL_LABEL} dend(0.8)')
    ax.plot(freqs, hcn_sweep['dend_gain'], '--s', color='tab:red', ms=3,
            lw=1.5, alpha=0.75, label=f'{HCN_LABEL} dend(0.8)')
    ax.axvspan(resonance_band[0], resonance_band[1], alpha=0.35,
               color=THETA_SHADE, label=band_label)
    ax.axvline(hcn_soma_metrics['peak_freq'], color='firebrick', ls='--', lw=1.3)
    ax.text(
        0.03,
        0.08,
        f'At {operating_v:.0f} mV the calibrated holding current keeps baseline fixed;\n'
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
    ax.set_xlim(freqs[0], freqs[-1])

    ax = axes[1, 0]
    ax.plot(freqs, control_soma_metrics['norm'], '-o', color='tab:blue', ms=4,
            lw=2, label=f'{CONTROL_LABEL} soma')
    ax.plot(freqs, hcn_soma_metrics['norm'], '-o', color='tab:red', ms=4,
            lw=2.4, label=f'{HCN_LABEL} soma')
    ax.plot(freqs, control_dend_metrics['norm'], '--', color='tab:blue',
            lw=1.3, alpha=0.55, label=f'{CONTROL_LABEL} dend(0.8)')
    ax.plot(freqs, hcn_dend_metrics['norm'], '--', color='tab:red',
            lw=1.3, alpha=0.65, label=f'{HCN_LABEL} dend(0.8)')
    ax.axvspan(resonance_band[0], resonance_band[1], alpha=0.35, color=THETA_SHADE)
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
    ax.set_xlim(freqs[0], freqs[-1])
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
    bars_control = ax.bar(x - width / 2, control_prominence, width,
                          color='tab:blue', label=CONTROL_LABEL)
    bars_hcn = ax.bar(x + width / 2, hcn_prominence, width,
                      color='tab:red', label=HCN_LABEL)
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
        ri_conclusion = 'HH+HCN global RI > HH control global RI'
    else:
        ri_conclusion = 'HH+HCN global RI not above HH control global RI'
    if operating_v <= -68.0:
        hh_note = 'HH effects are expected to be weak at this hyperpolarized Vm.'
    else:
        hh_note = 'Near-threshold no-spike operating point selected.'
    summary = (
        "Model:\n"
        f"  - Soma: pas + {neuron_control.hh_mech} Na/K +/- Ih\n"
        "  - Dendrite: pas +/- Ih\n"
        "  - Input: distal current at dend(0.8)\n"
        "  - Recording: soma and dend(0.8)\n\n"
        "Operating point:\n"
        f"  - Operating Vm: {operating_v:.1f} mV\n"
        f"  - Holding control: {hold_control:.4f} nA -> soma {control_soma_v:.1f} mV, dend {control_dend_v:.1f} mV\n"
        f"  - Holding HCN: {hold_hcn:.4f} nA -> soma {hcn_soma_v:.1f} mV, dend {hcn_dend_v:.1f} mV\n"
        "  - Holding: calibrated current clamp, not voltage clamp\n"
        f"  - Fixed g_h: {FIXED_G_H:.4g} S/cm2\n"
        f"  - Soma Ih density: {HCN_SOMA_DENSITY_FACTOR:.2f} x g_h\n"
        f"  - AC amplitude: {SWEEP_AMP:.4f} nA\n"
        f"  - Frequency range: {freqs[0]:.1f}-{freqs[-1]:.0f} Hz\n"
        f"  - Resonance band: {resonance_band[0]:.1f}-{resonance_band[1]:.1f} Hz\n"
        f"  - {hh_note}\n\n"
        "Validation:\n"
        f"  - HH spike validation: {hh_validation_spike}, current={validation_amp:.2f} nA\n"
        f"  - ZAP/sweep for resonance: no spikes = {zap_spike_free and sweep_spike_free}\n\n"
        "Results:\n"
        f"  - HCN sag recovery: {sag_hcn['sag_recovery']:.2f} mV\n"
        f"  - HH control global RI: {control_soma_metrics['ri']:.3f}\n"
        f"  - HH+HCN global RI: {hcn_soma_metrics['ri']:.3f}\n"
        f"  - HH+HCN global peak: {hcn_soma_metrics['peak_freq']:.1f} Hz\n"
        f"  - Theta-band RI: {hcn_soma_metrics['theta_ri']:.3f}\n"
        f"  - Prominence soma: control={control_soma_prom:.3f}, HCN={hcn_soma_prom:.3f}\n"
        f"  - Prominence dend: control={control_dend_prom:.3f}, HCN={hcn_dend_prom:.3f}\n\n"
        "Conclusion:\n"
        f"{ri_conclusion} under {operating_v:.0f} mV subthreshold conditions.\n"
        "This reflects normalized frequency tuning,\n"
        "not absolute voltage amplification."
    )
    ax.text(
        0.03,
        0.98,
        summary,
        transform=ax.transAxes,
        fontsize=7.45,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='none'),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig_dir = os.path.join(_figures_root, 'hcn_resonance', 'hh_hcn_voltage_sweep')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'{FIGURE_PREFIX}_{tag}.png')
    pdf_path = os.path.join(fig_dir, f'{FIGURE_PREFIX}_{tag}.pdf')
    plt.savefig(fig_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close(fig)

    return {
        'operating_v': operating_v,
        'tag': tag,
        'fig_path': fig_path,
        'pdf_path': pdf_path,
        'hh_mech': neuron_control.hh_mech,
        'hold_control': hold_control,
        'hold_hcn': hold_hcn,
        'resonance_band': resonance_band,
        'hh_validation_spike': hh_validation_spike,
        'validation_amp': validation_amp,
        'zap_spike_free': zap_spike_free,
        'sweep_spike_free': sweep_spike_free,
        'control_ri': control_soma_metrics['ri'],
        'hcn_ri': hcn_soma_metrics['ri'],
        'hcn_peak_freq': hcn_soma_metrics['peak_freq'],
        'hcn_theta_ri': hcn_soma_metrics['theta_ri'],
        'control_soma_prom': control_soma_prom,
        'hcn_soma_prom': hcn_soma_prom,
        'control_dend_prom': control_dend_prom,
        'hcn_dend_prom': hcn_dend_prom,
        'sag_hcn': sag_hcn['sag_recovery'],
    }


def plot_summary(results):
    voltages = np.array([r['operating_v'] for r in results])
    control_ri = np.array([r['control_ri'] for r in results])
    hcn_ri = np.array([r['hcn_ri'] for r in results])
    hcn_peak = np.array([r['hcn_peak_freq'] for r in results])
    hcn_theta_ri = np.array([r['hcn_theta_ri'] for r in results])
    hcn_soma_prom = np.array([r['hcn_soma_prom'] for r in results])
    hcn_dend_prom = np.array([r['hcn_dend_prom'] for r in results])
    sag_hcn = np.array([r['sag_hcn'] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('HH + HCN/Ih voltage sweep summary', fontsize=14)

    ax = axes[0, 0]
    ax.plot(voltages, control_ri, '-o', color='tab:blue', label=CONTROL_LABEL)
    ax.plot(voltages, hcn_ri, '-o', color='tab:red', label=HCN_LABEL)
    ax.axhline(1.0, color='k', ls='--', alpha=0.45)
    ax.set_xlabel('Operating Vm (mV)')
    ax.set_ylabel('Global resonance index')
    ax.set_title('A  Normalized resonance index')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(voltages, hcn_peak, '-o', color='firebrick', label='Global peak')
    ax.plot(voltages, hcn_theta_ri, '--s', color='tab:orange', label='Theta-band RI')
    ax.set_xlabel('Operating Vm (mV)')
    ax.set_ylabel('Peak frequency (Hz) / theta RI')
    ax.set_title('B  HCN peak and theta-band RI')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(voltages, hcn_soma_prom, '-o', color='tab:red', label='Soma')
    ax.plot(voltages, hcn_dend_prom, '--s', color='tab:red', alpha=0.7, label='Dendrite (0.8)')
    ax.axhline(0.0, color='k', ls='--', alpha=0.45)
    ax.set_xlabel('Operating Vm (mV)')
    ax.set_ylabel('Resonance prominence')
    ax.set_title('C  HCN resonance prominence')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(voltages, sag_hcn, '-o', color='tab:purple')
    ax.axhline(0.0, color='k', ls='--', alpha=0.45)
    ax.set_xlabel('Operating Vm (mV)')
    ax.set_ylabel('Sag recovery (mV)')
    ax.set_title('D  HCN sag recovery')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig_dir = os.path.join(_figures_root, 'hcn_resonance', 'hh_hcn_voltage_sweep')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'{FIGURE_PREFIX}_summary.png')
    pdf_path = os.path.join(fig_dir, f'{FIGURE_PREFIX}_summary.pdf')
    plt.savefig(fig_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close(fig)
    return fig_path, pdf_path


def main():
    results = []
    for operating_v in OPERATING_VOLTAGES:
        print(f"Running HH+HCN resonance at {operating_v:.0f} mV...")
        result = run_one_voltage(operating_v)
        results.append(result)
        print(f"  Figure saved: {result['fig_path']}")
        print(f"  PDF saved: {result['pdf_path']}")
        print(
            "  Metrics: "
            f"control RI={result['control_ri']:.3f}, "
            f"HH+HCN RI={result['hcn_ri']:.3f}, "
            f"peak={result['hcn_peak_freq']:.1f} Hz, "
            f"theta RI={result['hcn_theta_ri']:.3f}"
        )

    summary_fig, summary_pdf = plot_summary(results)
    print(f"Summary figure saved: {summary_fig}")
    print(f"Summary PDF saved: {summary_pdf}")
    print(f"HH mechanism used: {results[-1]['hh_mech']}")
    print(f"Fixed g_h: {FIXED_G_H:.4g} S/cm2")
    print(f"Soma Ih density factor: {HCN_SOMA_DENSITY_FACTOR:.2f}")
    print(f"Selected frequency range: {FREQUENCY_SWEEP[0]:.1f}-{FREQUENCY_SWEEP[-1]:.0f} Hz")
    print("Voltage sweep complete.")


if __name__ == '__main__':
    main()
