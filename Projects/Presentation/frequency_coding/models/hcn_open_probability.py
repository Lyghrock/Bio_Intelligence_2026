"""
HCN/Ih Open Probability vs Membrane Voltage
===========================================
Plots the steady-state open probability used by the current Ih.mod mechanism.

The active HCN conductance in Ih.mod is:
    gIh = gIhbar * m

Therefore the channel open probability is represented by the gate variable m.
At steady state, open probability is mInf.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_figures_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))

REFERENCE_MOD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'reference_mod', 'Ih.mod')
)

VOLTAGE_MIN = -140.0
VOLTAGE_MAX = -40.0
VOLTAGE_STEP = 0.1
OPERATING_POINTS = {
    'HH+HCN -90 mV': -90.0,
    'HCN -75 mV': -75.0,
    'HH+HCN -65 mV': -65.0,
}
POINT_COLORS = {
    'HH+HCN -90 mV': 'tab:red',
    'HCN -75 mV': 'tab:orange',
    'HH+HCN -65 mV': 'tab:green',
}


def hcn_rates(v_mV):
    """Return alpha, beta, mInf, and mTau exactly as defined in Ih.mod."""
    v = np.asarray(v_mV, dtype=float).copy()
    singular = np.isclose(v, -154.9)
    v[singular] += 0.0001

    m_alpha = 0.001 * 6.43 * (v + 154.9) / (np.exp((v + 154.9) / 11.9) - 1)
    m_beta = 0.001 * 193 * np.exp(v / 33.1)
    m_inf = m_alpha / (m_alpha + m_beta)
    m_tau = 1 / (m_alpha + m_beta)
    return m_alpha, m_beta, m_inf, m_tau


def hcn_open_probability(v_mV):
    """Steady-state HCN open probability, mInf."""
    _, _, m_inf, _ = hcn_rates(v_mV)
    return m_inf


def main():
    voltages = np.arange(VOLTAGE_MIN, VOLTAGE_MAX + VOLTAGE_STEP, VOLTAGE_STEP)
    open_probability = hcn_open_probability(voltages)
    _, _, _, tau = hcn_rates(voltages)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle('HCN/Ih gating from the current Ih.mod mechanism', fontsize=14)

    ax = axes[0]
    ax.plot(voltages, open_probability, color='tab:red', lw=2.5)
    open_summary = []
    for label, voltage in OPERATING_POINTS.items():
        prob = float(hcn_open_probability([voltage])[0])
        color = POINT_COLORS[label]
        ax.axvline(voltage, color=color, lw=1.2, ls='--', alpha=0.7)
        ax.plot(voltage, prob, 'o', color=color, ms=6)
        open_summary.append(f'{label}: {prob:.4f}')
    ax.text(
        0.58,
        0.94,
        'Operating-point Popen\n' + '\n'.join(open_summary),
        transform=ax.transAxes,
        fontsize=8.5,
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='0.8'),
    )
    ax.set_xlabel('Membrane voltage (mV)')
    ax.set_ylabel('Open probability (mInf)')
    ax.set_title('Steady-state open probability')
    ax.set_xlim(VOLTAGE_MIN, VOLTAGE_MAX)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(voltages, tau, color='tab:blue', lw=2.5)
    tau_summary = []
    for label, voltage in OPERATING_POINTS.items():
        tau_value = float(hcn_rates([voltage])[3][0])
        color = POINT_COLORS[label]
        ax.axvline(voltage, color=color, lw=1.2, ls='--', alpha=0.7)
        ax.plot(voltage, tau_value, 'o', color=color, ms=6)
        tau_summary.append(f'{label}: {tau_value:.1f} ms')
    ax.text(
        0.58,
        0.94,
        'Operating-point tau\n' + '\n'.join(tau_summary),
        transform=ax.transAxes,
        fontsize=8.5,
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='0.8'),
    )
    ax.set_xlabel('Membrane voltage (mV)')
    ax.set_ylabel('Activation time constant (ms)')
    ax.set_title('Activation time constant')
    ax.set_xlim(VOLTAGE_MIN, VOLTAGE_MAX)
    ax.grid(True, alpha=0.3)

    formula = (
        'Ih.mod parameters:\n'
        'mAlpha = 0.001 * 6.43 * (v + 154.9) / (exp((v + 154.9) / 11.9) - 1)\n'
        'mBeta  = 0.001 * 193 * exp(v / 33.1)\n'
        'Popen = mInf = mAlpha / (mAlpha + mBeta)\n'
        'mTau = 1 / (mAlpha + mBeta)\n'
        'Current resonance scripts set ehcn=-45 mV and gIhbar by location; these scale current/conductance, not Popen.'
    )
    fig.text(
        0.02,
        0.01,
        formula,
        fontsize=8.5,
        fontfamily='monospace',
        va='bottom',
    )

    plt.tight_layout(rect=(0, 0.18, 1, 0.93))
    fig_dir = os.path.join(_figures_root, 'hcn_resonance')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'hcn_open_probability_vs_voltage.png')
    pdf_path = os.path.join(fig_dir, 'hcn_open_probability_vs_voltage.pdf')
    plt.savefig(fig_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close(fig)

    print(f"Ih.mod source: {REFERENCE_MOD}")
    print(f"Figure saved: {fig_path}")
    print(f"PDF saved: {pdf_path}")
    for label, voltage in OPERATING_POINTS.items():
        prob = float(hcn_open_probability([voltage])[0])
        tau_value = float(hcn_rates([voltage])[3][0])
        print(f"{label}: V={voltage:.1f} mV, Popen={prob:.4f}, tau={tau_value:.2f} ms")
    print("Simulation complete.")


if __name__ == '__main__':
    main()
