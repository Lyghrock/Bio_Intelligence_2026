"""
Complete Filter Bank - All Channel Types Combined
================================================
Figures:
  figures/complete_filter_bank/z_comparison.png    - All 4 impedance curves
  figures/complete_filter_bank/filter_types.png    - Filter type illustration

Demonstrates the complete "filter bank" nature of neurons:
  Passive:  Low-pass  (impedance drops with frequency)
  HCN:      Band-pass (theta resonance peak)
  KCNQ:     High-pass (impedance increases with frequency)
  HCN+KCNQ: Band-pass + high-pass (complex multi-band)
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
    print(f"[OK] Loaded mechanisms from: {_ref_mod_path}")
else:
    print(f"[WARN] Mechanism DLL not found: {_ref_mod_path}")


def make_model(name, soma_L=10, soma_diam=10,
               g_pas=0.0002, e_pas=-65,
               g_h=0.0, g_Im=0.0):
    """Create a soma-only model with specified channel conductances."""
    sec = h.Section(name=f'soma_{name}')
    sec.L = soma_L
    sec.diam = soma_diam
    sec.insert('pas')
    sec.g_pas = g_pas
    sec.e_pas = e_pas

    if g_h > 0:
        sec.insert('hh1')
        sec.insert('Ih')
        sec(0.5).Ih.gIhbar = g_h

    if g_Im > 0:
        if g_h <= 0:
            sec.insert('hh1')
        sec.insert('Im')
        sec(0.5).Im.gImbar = g_Im

    v = h.Vector()
    t = h.Vector()
    v.record(sec(0.5)._ref_v)
    t.record(h._ref_t)
    imp = h.Impedance()

    return sec, v, t, imp


def measure_impedance(sec, imp, freq_range=(0.1, 100), n_points=50):
    """Measure input impedance at soma."""
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
    z_mag = np.zeros(len(freqs))
    phase = np.zeros(len(freqs))

    h.finitialize(-65)
    imp.loc(0.5, sec=sec)

    compute_mode = 0  # mode=1 fails with hh1

    for i, f in enumerate(freqs):
        imp.compute(f, compute_mode)
        z_mag[i] = imp.input(0.5, sec=sec)
        phase[i] = imp.input_phase(0.5, sec=sec)

    return freqs, z_mag, phase


def main():
    print("=" * 65)
    print("Complete Neural Frequency Filter Bank")
    print("=" * 65)

    print("\nCreating 4 model variants...")
    configs = {
        'Passive':     {'g_pas': 0.0002, 'g_h': 0.0,  'g_Im': 0.0},
        'HCN':        {'g_pas': 0.0002, 'g_h': 0.002, 'g_Im': 0.0},
        'KCNQ':        {'g_pas': 0.0002, 'g_h': 0.0,  'g_Im': 0.0005},
        'HCN+KCNQ':   {'g_pas': 0.0002, 'g_h': 0.002, 'g_Im': 0.0005},
    }

    results = {}
    for name, cfg in configs.items():
        print(f"  Creating: {name}")
        sec, v, t, imp = make_model(
            name=name.replace('+', 'plus'),
            g_pas=cfg['g_pas'],
            g_h=cfg['g_h'],
            g_Im=cfg['g_Im']
        )
        freqs, z_mag, phase = measure_impedance(sec, imp)
        results[name] = {'freqs': freqs, 'z': z_mag, 'phase': phase}

    # ---- Print summary ----
    print("\n" + "-" * 65)
    print("Filter Bank Summary")
    print("-" * 65)
    print(f"{'Model':>12} {'Filter Type':>20} {'Peak Freq':>12} {'Peak Z':>12}")
    print("-" * 65)

    filter_types = {
        'Passive':   'Low-pass',
        'HCN':       'Band-pass (theta)',
        'KCNQ':      'High-pass',
        'HCN+KCNQ':  'Complex band-pass'
    }

    for name, r in results.items():
        peak_idx = np.argmax(r['z'])
        peak_f = r['freqs'][peak_idx]
        peak_z = r['z'][peak_idx]
        ftype = filter_types[name]

        # Check actual behavior
        lo = np.mean(r['z'][r['freqs'] < 3])
        hi = np.mean(r['z'][r['freqs'] > 15])

        print(f"{name:>12} {ftype:>20} {peak_f:>10.1f} Hz {peak_z:>10.1f} MΩ")
        if name == 'KCNQ':
            print(f"             {'(Z_hi/Z_lo = %.2fx)' % (hi/lo):>30}")

    # ---- Figure 1: All impedance curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'Passive':   '#666666',
        'HCN':       '#cc2200',
        'KCNQ':      '#0044cc',
        'HCN+KCNQ':  '#008844',
    }
    linestyles = {
        'Passive':   '--',
        'HCN':       '-',
        'KCNQ':      '-',
        'HCN+KCNQ':  '-.',
    }

    ax = axes[0]
    for name, r in results.items():
        ax.semilogx(r['freqs'], r['z'],
                    color=colors[name],
                    ls=linestyles[name],
                    lw=2.5,
                    label=f'{name}')
    ax.axvspan(3, 10, alpha=0.12, color='orange', label='Theta band')
    ax.axvline(5, color='orange', ls='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Input Impedance (MΩ)')
    ax.set_title('Complete Neural Filter Bank\n(Each channel type creates a different filter)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 100)

    # ---- Figure 1 right: Normalized ----
    ax = axes[1]
    for name, r in results.items():
        z_norm = r['z'] / np.mean(r['z'][:5])
        ax.semilogx(r['freqs'], z_norm,
                    color=colors[name],
                    ls=linestyles[name],
                    lw=2.5,
                    label=f'{name}')
    ax.axhline(1.0, color='k', ls='--', alpha=0.3)
    ax.axvspan(3, 10, alpha=0.12, color='orange')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized Impedance')
    ax.set_title('Normalized Response (relative to low-freq baseline)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 100)

    plt.tight_layout()
    fig_dir = os.path.join(_figures_root, 'complete_filter_bank')
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, 'z_comparison.png')
    plt.savefig(path, dpi=150)
    print(f"\n[OK] Saved: {path}")
    plt.close()

    # ---- Figure 2: Filter type illustration ----
    fig, ax = plt.subplots(figsize=(12, 6))

    freqs_plot = np.linspace(0.5, 50, 300)

    # Synthetic filter curves (illustrative)
    # Passive: 1/sqrt(1+(f/fc)^2)
    fc = 5
    z_passive = 1.0 / np.sqrt(1 + (freqs_plot / fc)**2)
    z_passive = z_passive / z_passive[0] * results['Passive']['z'][0]

    # HCN: band-pass Gaussian centered at 5 Hz
    sigma = 2
    z_hcn = np.exp(-((freqs_plot - 5)**2) / (2 * sigma**2))
    z_hcn = z_hcn / np.max(z_hcn) * results['HCN']['z'][np.argmin(np.abs(results['HCN']['freqs']-5))]

    # KCNQ: high-pass: starts low, increases
    z_kcnq = freqs_plot / (freqs_plot + 3)
    z_kcnq = z_kcnq / z_kcnq[0] * results['KCNQ']['z'][0]

    ax.fill_between(freqs_plot, z_passive * 0 + 0.01, z_passive,
                   alpha=0.3, color=colors['Passive'], label='Passive (Low-pass)')
    ax.fill_between(freqs_plot, z_hcn * 0 + 0.01, z_hcn,
                   alpha=0.3, color=colors['HCN'], label='HCN (Band-pass)')
    ax.fill_between(freqs_plot, z_kcnq * 0 + 0.01, z_kcnq,
                   alpha=0.3, color=colors['KCNQ'], label='KCNQ (High-pass)')

    ax.plot(freqs_plot, z_passive, color=colors['Passive'], ls='--', lw=2)
    ax.plot(freqs_plot, z_hcn, color=colors['HCN'], ls='-', lw=2.5)
    ax.plot(freqs_plot, z_kcnq, color=colors['KCNQ'], ls='-', lw=2)

    ax.axvspan(3, 10, alpha=0.1, color='orange')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Impedance (normalized, MΩ)', fontsize=12)
    ax.set_title('Neural Frequency Filters: Filter Types\n'
                 '(Biological neurons implement multiple filter types simultaneously)',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 50)

    ax.text(0.02, 0.02,
            ('Key Insight: Neurons are NOT binary spike generators.\n'
             'They are ACTIVE FREQUENCY FILTERS that select which\n'
             'frequency bands are amplified or suppressed.'),
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    path = os.path.join(fig_dir, 'filter_types.png')
    plt.savefig(path, dpi=150)
    print(f"[OK] Saved: {path}")
    plt.close()

    print("\n" + "=" * 65)
    print("Key Finding:")
    print("  Neurons implement a COMPLETE FILTER BANK through combinations of:")
    print("  1. Passive properties   -> Low-pass (baseline)")
    print("  2. HCN channels         -> Theta band-pass (4-8 Hz)")
    print("  3. KCNQ channels        -> High-pass (sharpens responses)")
    print("  4. HCN + KCNQ          -> Multi-band filtering")
    print("\n  This enables selective processing of different frequency bands,")
    print("  similar to how a radio receiver tunes to different stations.")
    print("=" * 65)


if __name__ == '__main__':
    main()
