"""
HCN Neuron Model - Resonance Analysis
=====================================
Hyperpolarized subthreshold HH + HCN/Ih simulation showing that HCN/Ih
changes normalized membrane frequency tuning.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hcn_resonance_common import (
    DEFAULT_EH_HCN,
    DEFAULT_G_H,
    FREQUENCY_SWEEP_THETA,
    HCN_SOMA_DENSITY_FACTOR,
    HHHCNResonanceNeuron,
    THETA_SHADE,
    baseline_is_stable,
    compute_step_metrics,
    detected_resonance_band,
    frequency_response_metrics,
    plot_zap_trace,
    resonance_band_label,
    resonance_prominence,
    run_frequency_sweep,
    set_tight_ylim,
)


_figures_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))

CONTROL_LABEL = 'No HCN control'
HCN_LABEL = 'With HCN'
OPERATING_V = -75.0
FIXED_G_H = DEFAULT_G_H
SIM_DT = 0.5
ZAP_AMP = 0.01
SWEEP_AMP = 0.01
STEP_AMP = -0.08
FREQUENCY_SWEEP = FREQUENCY_SWEEP_THETA


def main():
    freqs = FREQUENCY_SWEEP

    neuron_control = HHHCNResonanceNeuron(
        'hcn_control',
        with_hcn=False,
        dend_L=300,
        g_h=FIXED_G_H,
        g_pas_dend=0.0003,
        eh_hcn=DEFAULT_EH_HCN,
        hcn_soma_density=HCN_SOMA_DENSITY_FACTOR,
        sim_dt=SIM_DT,
    )
    neuron_hcn = HHHCNResonanceNeuron(
        'hcn_with_ih',
        with_hcn=True,
        dend_L=300,
        g_h=FIXED_G_H,
        g_pas_dend=0.0003,
        eh_hcn=DEFAULT_EH_HCN,
        hcn_soma_density=HCN_SOMA_DENSITY_FACTOR,
        sim_dt=SIM_DT,
    )

    hold_control, _, _ = neuron_control.calibrate_holding_current(operating_v=OPERATING_V)
    hold_hcn, _, _ = neuron_hcn.calibrate_holding_current(operating_v=OPERATING_V)
    if not baseline_is_stable(neuron_control, hold_control, OPERATING_V):
        raise RuntimeError(f"Control baseline unstable at {OPERATING_V:.1f} mV")
    if not baseline_is_stable(neuron_hcn, hold_hcn, OPERATING_V):
        raise RuntimeError(f"HCN baseline unstable at {OPERATING_V:.1f} mV")

    control_sweep = run_frequency_sweep(
        neuron_control,
        freqs,
        hold_control,
        ac_amp=SWEEP_AMP,
        operating_v=OPERATING_V,
    )
    hcn_sweep = run_frequency_sweep(
        neuron_hcn,
        freqs,
        hold_hcn,
        ac_amp=SWEEP_AMP,
        operating_v=OPERATING_V,
    )
    control_soma_metrics = frequency_response_metrics(freqs, control_sweep['soma_gain'])
    control_dend_metrics = frequency_response_metrics(freqs, control_sweep['dend_gain'])
    hcn_soma_metrics = frequency_response_metrics(freqs, hcn_sweep['soma_gain'])
    hcn_dend_metrics = frequency_response_metrics(freqs, hcn_sweep['dend_gain'])

    resonance_band = detected_resonance_band(freqs, hcn_soma_metrics['peak_freq'])
    band_label = resonance_band_label(resonance_band)
    prom_control_soma = resonance_prominence(freqs, control_sweep['soma_gain'], resonance_band)
    prom_hcn_soma = resonance_prominence(freqs, hcn_sweep['soma_gain'], resonance_band)
    prom_control_dend = resonance_prominence(freqs, control_sweep['dend_gain'], resonance_band)
    prom_hcn_dend = resonance_prominence(freqs, hcn_sweep['dend_gain'], resonance_band)

    hh_validation_spike, validation_amp, _, _ = neuron_control.validate_hh_spiking()

    t_control_sag, v_control_sag, _ = neuron_control.hyperpolarization_test(
        hold_amp=hold_control,
        operating_v=OPERATING_V,
        step_amp=STEP_AMP,
    )
    t_hcn_sag, v_hcn_sag, _ = neuron_hcn.hyperpolarization_test(
        hold_amp=hold_hcn,
        operating_v=OPERATING_V,
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
        operating_v=OPERATING_V,
    )
    t_hcn_zap, v_hcn_zap, _ = neuron_hcn.zap_trace(
        duration=4000,
        f_start=0.5,
        f_end=25,
        ac_amp=ZAP_AMP,
        hold_amp=hold_hcn,
        operating_v=OPERATING_V,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'HCN/Ih reshapes subthreshold membrane frequency tuning toward resonance',
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

    plot_zap_trace(
        axes[0, 1],
        t_control_zap,
        v_control_zap,
        t_hcn_zap,
        v_hcn_zap,
        CONTROL_LABEL,
        HCN_LABEL,
        'Subthreshold ZAP trace, no spikes',
    )

    ax = axes[0, 2]
    ax.plot(freqs, control_sweep['soma_gain'], '-o', color='tab:blue', ms=4, lw=2,
            label=f'{CONTROL_LABEL} soma')
    ax.plot(freqs, hcn_sweep['soma_gain'], '-o', color='tab:red', ms=4, lw=2,
            label=f'{HCN_LABEL} soma')
    ax.plot(freqs, control_sweep['dend_gain'], '--s', color='tab:blue', ms=3,
            lw=1.5, alpha=0.65, label=f'{CONTROL_LABEL} dend(0.8)')
    ax.plot(freqs, hcn_sweep['dend_gain'], '--s', color='tab:red', ms=3,
            lw=1.5, alpha=0.75, label=f'{HCN_LABEL} dend(0.8)')
    ax.axvspan(resonance_band[0], resonance_band[1], alpha=0.35,
               color=THETA_SHADE, label=band_label)
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
        'but resonance is assessed from normalized frequency tuning.',
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='none'),
    )
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Transfer Gain (MOhm)')
    ax.set_title('Absolute Transfer Gain')
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
        f'HCN global RI = {hcn_soma_metrics["ri"]:.2f}\n'
        f'Control global RI = {control_soma_metrics["ri"]:.2f}\n'
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
    control_prominence = np.array([prom_control_soma, prom_control_dend])
    hcn_prominence = np.array([prom_hcn_soma, prom_hcn_dend])
    bars_control = ax.bar(x - width / 2, control_prominence, width,
                          color='tab:blue', label=CONTROL_LABEL)
    bars_hcn = ax.bar(x + width / 2, hcn_prominence, width,
                      color='tab:red', label=HCN_LABEL)
    ax.axhline(0.0, color='k', ls='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Soma', 'Dendrite (0.8)'])
    ax.set_ylabel('Resonance prominence')
    ax.set_title('Resonance prominence')
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
        'HCN global RI > control global RI'
        if hcn_soma_metrics['ri'] > control_soma_metrics['ri']
        else 'HCN global RI not above control global RI'
    )
    prom_statement = (
        'HCN > control at soma and dendrite'
        if prom_hcn_soma > prom_control_soma and prom_hcn_dend > prom_control_dend
        else 'see prominence panel'
    )
    summary = (
        "Model:\n"
        f"  - Soma: pas + {neuron_control.hh_mech} Na/K +/- Ih\n"
        "  - Dendrite: pas +/- Ih\n"
        "  - Input: distal current at dend(0.8)\n"
        "  - Recording: soma and dend(0.8)\n\n"
        "Operating point:\n"
        f"  - Operating Vm: {OPERATING_V:.1f} mV\n"
        "  - Holding: current clamp, not voltage clamp\n"
        f"  - Fixed g_h: {FIXED_G_H:.4g} S/cm2\n"
        f"  - Soma Ih density: {HCN_SOMA_DENSITY_FACTOR:.2f} x g_h\n"
        f"  - Resonance band: {resonance_band[0]:.1f}-{resonance_band[1]:.1f} Hz\n\n"
        "Validation:\n"
        f"  - HH spike validation: {hh_validation_spike}, current={validation_amp:.2f} nA\n"
        "  - ZAP/sweep for resonance: no spikes\n\n"
        "Results:\n"
        f"  - Sag recovery: control={sag_control['sag_recovery']:.2f} mV, "
        f"HCN={sag_hcn['sag_recovery']:.2f} mV\n"
        "  - Absolute gain can be lower with HCN\n"
        "    because Ih increases membrane conductance\n"
        f"  - Normalized impedance: {ri_statement}\n"
        f"  - HCN global peak: {hcn_soma_metrics['peak_freq']:.1f} Hz\n"
        f"  - Resonance prominence: {prom_statement}\n\n"
        "Conclusion:\n"
        "HCN/Ih produces relative resonance under\n"
        "hyperpolarized subthreshold conditions.\n"
        "This is a normalized frequency-tuning effect,\n"
        "not an absolute voltage-amplitude amplification."
    )
    ax.text(
        0.03,
        0.98,
        summary,
        transform=ax.transAxes,
        fontsize=8.2,
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
    print(f"HH mechanism used: {neuron_control.hh_mech}")
    print(f"Operating Vm: {OPERATING_V:.1f} mV")
    print(f"Fixed g_h: {FIXED_G_H:.4g} S/cm2")
    print(f"Soma Ih density factor: {HCN_SOMA_DENSITY_FACTOR:.2f}")
    print(f"Resonance band: {resonance_band[0]:.1f}-{resonance_band[1]:.1f} Hz")
    print(f"HH validation spike: {hh_validation_spike}, current={validation_amp:.2f} nA")
    print(f"Sag recovery: control={sag_control['sag_recovery']:.2f} mV, "
          f"HCN={sag_hcn['sag_recovery']:.2f} mV")
    print(f"Control global RI: {control_soma_metrics['ri']:.3f}")
    print(f"HCN global RI: {hcn_soma_metrics['ri']:.3f}")
    print(f"HCN global peak frequency: {hcn_soma_metrics['peak_freq']:.1f} Hz")
    print("Resonance prominence: "
          f"control soma={prom_control_soma:.3f}, "
          f"HCN soma={prom_hcn_soma:.3f}, "
          f"control dend={prom_control_dend:.3f}, "
          f"HCN dend={prom_hcn_dend:.3f}")
    print("Simulation complete.")


if __name__ == '__main__':
    main()
