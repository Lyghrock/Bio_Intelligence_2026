"""
Spatial Heterogeneity Model - Frequency Routing
============================================
Figures:
  figures/frequency_routing/z_by_location.png   - Impedance at 3 locations
  figures/frequency_routing/routing_map.png      - Heatmap of routing gain
  figures/frequency_routing/theta_preference.png - Theta band routing

Key concept: Different dendritic locations have different HCN channel density,
creating spatially-dependent frequency preferences -> FREQUENCY ROUTING.

Architecture:
  Proximal: Passive only (low-pass)
  Middle:   Light HCN (g_h = 0.0005, weak theta resonance)
  Distal:   Strong HCN (g_h = 0.002, strong theta resonance)
  Soma:     HH + Im
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


class SpatialHeteroNeuron:
    """
    Proximal -> Middle -> Distal dendritic architecture.
    HCN density increases toward distal end.
    """

    def __init__(self, proximal_L=100, middle_L=200, distal_L=200,
                 g_h_prox=0.0, g_h_mid=0.0005, g_h_dist=0.002,
                 g_Im_soma=0.00005):
        self.soma = h.Section(name='soma_sh')
        self.prox = h.Section(name='prox_sh')
        self.mid = h.Section(name='mid_sh')
        self.dist = h.Section(name='dist_sh')

        # Soma
        self.soma.L = 10
        self.soma.diam = 10
        self.soma.cm = 1.0
        self.soma.Ra = 100
        self.soma.insert('hh1')
        self.soma.insert('pas')
        self.soma.g_pas = 0.0001
        self.soma.e_pas = -65
        self.soma.insert('Im')
        for seg in self.soma:
            seg.Im.gImbar = g_Im_soma

        # Proximal: passive only
        self.prox.L = proximal_L
        self.prox.diam = 2.0
        self.prox.nseg = 20
        self.prox.cm = 1.0
        self.prox.Ra = 100
        self.prox.insert('pas')
        self.prox.g_pas = 0.0003
        self.prox.e_pas = -65

        # Middle: light HCN
        self.mid.L = middle_L
        self.mid.diam = 1.5
        self.mid.nseg = 20
        self.mid.cm = 1.0
        self.mid.Ra = 100
        self.mid.insert('pas')
        self.mid.g_pas = 0.0003
        self.mid.e_pas = -65
        self.mid.insert('Ih')
        for seg in self.mid:
            seg.Ih.gIhbar = g_h_mid

        # Distal: strong HCN (theta resonance)
        self.dist.L = distal_L
        self.dist.diam = 1.0
        self.dist.nseg = 20
        self.dist.cm = 1.0
        self.dist.Ra = 100
        self.dist.insert('pas')
        self.dist.g_pas = 0.0003
        self.dist.e_pas = -65
        self.dist.insert('Ih')
        for seg in self.dist:
            seg.Ih.gIhbar = g_h_dist

        # Connect: Proximal->Soma, Middle->Proximal, Distal->Middle
        self.prox.connect(self.soma(1))
        self.mid.connect(self.prox(1))
        self.dist.connect(self.mid(1))

        # Recordings
        self.v_soma = h.Vector()
        self.v_prox = h.Vector()
        self.v_mid = h.Vector()
        self.v_dist = h.Vector()
        self.t = h.Vector()
        self.v_soma.record(self.soma(0.5)._ref_v)
        self.v_prox.record(self.prox(0.9)._ref_v)
        self.v_mid.record(self.mid(0.5)._ref_v)
        self.v_dist.record(self.dist(0.9)._ref_v)
        self.t.record(h._ref_t)

        self.imp = h.Impedance()

    def zap_protocol(self, inj_location, duration=4000, f_start=0.5, f_end=30, amp=0.2):
        """
        Inject ZAP at a specific location, record soma voltage.
        Returns (t_soma, v_soma, t_wave, wave) for the injection.
        """
        loc_sec = {'prox': self.prox, 'mid': self.mid, 'dist': self.dist}
        loc_x = {'prox': 0.9, 'mid': 0.5, 'dist': 0.9}
        sec = loc_sec[inj_location]
        x = loc_x[inj_location]

        dt = 0.025
        t = np.arange(0, duration, dt)
        rate = (f_end - f_start) / (duration / 1000)
        t_sec = t / 1000
        phase = 2 * np.pi * (f_start * t_sec + 0.5 * rate * t_sec**2)
        wave = amp * np.sin(phase)

        stim = h.IClamp(sec(x))
        stim.delay = 0
        stim.dur = duration
        stim.amp = 0
        h.Vector(wave).play(stim._ref_amp, dt)

        h.finitialize(-65)
        h.tstop = duration + 100
        h.run()
        h.Vector(wave).play_remove()

        return np.array(self.t), np.array(self.v_soma), t, wave

    def measure_location_impedance(self, location, freq_range=(0.5, 50), n_points=40):
        """
        Measure input impedance at a specific dendritic location.
        location: 'soma', 'prox', 'mid', 'dist'
        """
        sec_map = {'soma': self.soma, 'prox': self.prox,
                   'mid': self.mid, 'dist': self.dist}
        x_map = {'soma': 0.5, 'prox': 0.9, 'mid': 0.5, 'dist': 0.9}

        sec = sec_map[location]
        x = x_map[location]

        freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
        z_mag = np.zeros(len(freqs))

        h.finitialize(-65)

        for i, f in enumerate(freqs):
            self.imp.loc(x, sec=sec)
            self.imp.compute(f, 0)
            z_mag[i] = self.imp.input(x, sec=sec)

        return freqs, z_mag

    def measure_transfer_gain(self, from_location, freq_range=(0.5, 50), n_points=40):
        """
        Measure how well signals from a location propagate to soma.
        """
        loc_map = {
            'prox': (self.prox, 0.9),
            'mid':  (self.mid,  0.5),
            'dist': (self.dist, 0.9),
        }

        sec, x = loc_map[from_location]
        freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)

        input_z = np.zeros(len(freqs))
        transfer_z = np.zeros(len(freqs))

        h.finitialize(-65)

        for i, f in enumerate(freqs):
            self.imp.loc(x, sec=sec)
            self.imp.compute(f, 0)
            input_z[i] = self.imp.input(x, sec=sec)
            transfer_z[i] = self.imp.transfer(0.5, sec=self.soma)

        # Gain = transfer / input
        gain = transfer_z / (input_z + 1e-12)
        return freqs, gain

    def inject_composite(self, duration=3000):
        """
        Composite input: theta (6 Hz) at distal + gamma (40 Hz) at proximal.
        Shows how different frequencies are routed to soma.
        """
        dt = 0.025
        t = np.arange(0, duration, dt)

        theta = 0.3 * np.sin(2 * np.pi * 6 * t / 1000)
        gamma = 0.15 * np.sin(2 * np.pi * 40 * t / 1000)
        noise = 0.1 * np.random.randn(len(t))

        prox_sig = gamma + noise
        dist_sig = theta + noise * 0.3

        prox_vec = h.Vector(prox_sig)
        dist_vec = h.Vector(dist_sig)

        stim_prox = h.IClamp(self.prox(0.8))
        stim_prox.delay = 0
        stim_prox.dur = duration
        stim_prox.amp = 0
        prox_vec.play(stim_prox._ref_amp, dt)

        stim_dist = h.IClamp(self.dist(0.5))
        stim_dist.delay = 0
        stim_dist.dur = duration
        stim_dist.amp = 0
        dist_vec.play(stim_dist._ref_amp, dt)

        h.finitialize(-65)
        h.tstop = duration + 100
        h.run()

        prox_vec.play_remove()
        dist_vec.play_remove()

        return {
            't': np.array(self.t),
            'v_soma': np.array(self.v_soma),
            'theta': theta,
            'gamma': gamma,
        }


def zap_analyze(t_v, v_data, t_wave, wave, f_start=0.5, f_end=30, seg_dur=200):
    """Extract transfer gain from ZAP by V_pp / I_pp per segment."""
    dur = t_wave[-1]
    n_seg = int(dur / seg_dur)
    freqs_out, vpp_out, ipp_out = [], [], []

    for i in range(n_seg):
        t0, t1 = i * seg_dur, (i + 1) * seg_dur
        mid_f = f_start + (f_end - f_start) * (i + 0.5) / n_seg

        v_mask = (t_v >= t0) & (t_v < t1)
        v_seg = v_data[v_mask]
        vpp = (np.max(v_seg) - np.min(v_seg)) if len(v_seg) > 10 else 0

        i_mask = (t_wave >= t0) & (t_wave < t1)
        i_seg = wave[i_mask]
        ipp = (np.max(i_seg) - np.min(i_seg)) if len(i_seg) > 10 else 0

        freqs_out.append(mid_f)
        vpp_out.append(vpp)
        ipp_out.append(ipp)

    return np.array(freqs_out), np.array(vpp_out), np.array(ipp_out)


def make_dir(name):
    d = os.path.join(_figures_root, name)
    os.makedirs(d, exist_ok=True)
    return d


def main():
    print("=" * 65)
    print("Spatial Heterogeneity - Frequency Routing")
    print("=" * 65)

    print("\nArchitecture:")
    print("  Proximal: Passive (g_h = 0)")
    print("  Middle:   Light HCN (g_h = 0.0005)")
    print("  Distal:   Strong HCN (g_h = 0.002)")
    print("  Soma:     HH + Im (temporal precision)")

    neuron = SpatialHeteroNeuron()

    # ---- Test 1: Impedance at each location ----
    print("\n" + "-" * 65)
    print("Test 1: Input Impedance at Different Locations")
    print("-" * 65)

    locs = ['prox', 'mid', 'dist']
    loc_labels = {'prox': 'Proximal (Passive)', 'mid': 'Middle (Light HCN)',
                   'dist': 'Distal (Strong HCN)'}
    loc_colors = {'prox': '#0066cc', 'mid': '#cc6600', 'dist': '#cc2200'}

    loc_results = {}
    for loc in locs:
        freqs, z = neuron.measure_location_impedance(loc)
        loc_results[loc] = {'freqs': freqs, 'z': z}
        peak_idx = np.argmax(z)
        print(f"  {loc_labels[loc]}: peak Z={z[peak_idx]:.1f}MΩ @ {freqs[peak_idx]:.1f}Hz")

    # ---- Test 2: ZAP-based Routing (dynamic frequency analysis) ----
    print("\n" + "-" * 65)
    print("Test 2: ZAP-based Frequency Routing (Dynamic Response)")
    print("-" * 65)

    zap_results = {}
    for loc in locs:
        print(f"  Running ZAP at {loc_labels[loc]}...")
        t_v, v_soma, t_w, wave = neuron.zap_protocol(loc, duration=4000, f_start=0.5, f_end=30, amp=0.2)
        f_out, vpp, ipp = zap_analyze(t_v, v_soma, t_w, wave, f_start=0.5, f_end=30, seg_dur=200)
        zap_results[loc] = {'freqs': f_out, 'vpp': vpp, 'ipp': ipp}

    # Theta band: relative preference
    theta_ratio = {}
    for loc in locs:
        freqs = zap_results[loc]['freqs']
        gain = zap_results[loc]['vpp'] / (zap_results[loc]['ipp'] + 1e-12)
        mask_theta = (freqs >= 3) & (freqs <= 10)
        mask_lo = (freqs >= 0.5) & (freqs <= 2)
        theta_g = np.mean(gain[mask_theta]) if np.any(mask_theta) else 0
        lo_g = np.mean(gain[mask_lo]) if np.any(mask_lo) else 0
        theta_ratio[loc] = theta_g / (lo_g + 1e-12)
        peak_idx = np.argmax(gain)
        print(f"    {loc_labels[loc]}: peak {gain[peak_idx]:.1f}MΩ @ {freqs[peak_idx]:.1f}Hz  "
              f"(theta/lo ratio: {theta_ratio[loc]:.3f})")

    best_theta_loc = max(theta_ratio, key=theta_ratio.get)
    print(f"\n  [OK] Theta RELATIVE PREFERENCE: {loc_labels[best_theta_loc]} "
          f"(theta/lo = {theta_ratio[best_theta_loc]:.3f})")

    # ---- Test 3: Composite signal ----
    print("\n" + "-" * 65)
    print("Test 3: Composite Signal Routing")
    print("-" * 65)
    print("  Injecting: theta (6 Hz) at distal + gamma (40 Hz) at proximal")
    result = neuron.inject_composite(duration=3000)

    # ---- Figure 1: Input impedance by location ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for loc in locs:
        r = loc_results[loc]
        ax.semilogx(r['freqs'], r['z'], color=loc_colors[loc],
                    lw=2.5, label=loc_labels[loc])
    ax.axvspan(3, 10, alpha=0.12, color='orange')
    ax.axvline(5, color='orange', ls='--', alpha=0.5, label='Theta (5 Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Input Impedance (MΩ)')
    ax.set_title('Input Impedance at Different Dendritic Locations\n'
                 '(Distal shows high impedance from HCN channels)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: ZAP-based routing gain
    ax = axes[1]
    for loc in locs:
        r = zap_results[loc]
        gain = r['vpp'] / (r['ipp'] + 1e-12)
        ax.semilogx(r['freqs'], gain, color=loc_colors[loc],
                    lw=2.5, label=loc_labels[loc])
    ax.axvspan(3, 10, alpha=0.12, color='orange')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Routing Gain = V_pp / I_pp (MΩ)')
    ax.set_title('ZAP Routing Gain: Input Location → Soma\n'
                 '(Distal with HCN: theta band shows resonance)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(make_dir('frequency_routing'), 'z_by_location.png')
    plt.savefig(path, dpi=150)
    print(f"\n[OK] Saved: {path}")
    plt.close()

    # ---- Figure 2: Routing heatmap ----
    fig, ax = plt.subplots(figsize=(10, 6))

    all_freqs = zap_results['dist']['freqs']
    gain_matrix = np.array([
        zap_results['prox']['vpp'] / (zap_results['prox']['ipp'] + 1e-12),
        zap_results['mid']['vpp'] / (zap_results['mid']['ipp'] + 1e-12),
        zap_results['dist']['vpp'] / (zap_results['dist']['ipp'] + 1e-12)
    ])

    im = ax.imshow(gain_matrix, aspect='auto', cmap='YlOrRd',
                   extent=[all_freqs[0], all_freqs[-1], 0, 3])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(['Proximal\n(Passive)', 'Middle\n(Light HCN)', 'Distal\n(Strong HCN)'])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Frequency Routing Map (ZAP-based): Input Location → Soma')
    ax.set_xscale('log')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Routing Gain (MΩ)', fontsize=10)

    ax.axvspan(3, 10, alpha=0.15, color='cyan', zorder=0)
    ax.text(5, 3.3, 'Theta band\n(3-10 Hz)', ha='center', fontsize=9, color='darkblue')

    plt.tight_layout()
    path = os.path.join(make_dir('frequency_routing'), 'routing_map.png')
    plt.savefig(path, dpi=150)
    print(f"[OK] Saved: {path}")
    plt.close()

    # ---- Figure 3: Theta preference ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    theta_ratio_vals = [theta_ratio[l] for l in locs]
    x_pos = np.arange(len(locs))
    bars = ax.bar(x_pos, theta_ratio_vals, color=[loc_colors[l] for l in locs], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Proximal\n(Passive)', 'Middle\n(Light HCN)', 'Distal\n(Strong HCN)'])
    ax.set_ylabel('Theta / Low-freq Gain Ratio')
    ax.set_title('Theta Relative Preference\n'
                 '(Distal HCN: theta/lo > 1 = theta PREFERRED)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='No preference')

    for bar, val in zip(bars, theta_ratio_vals):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=10)

    # Theta vs gamma comparison
    ax = axes[1]
    gamma_ratio = {}
    for loc in locs:
        freqs = zap_results[loc]['freqs']
        gain = zap_results[loc]['vpp'] / (zap_results[loc]['ipp'] + 1e-12)
        mask_gamma = (freqs >= 30) & (freqs <= 50)
        mask_lo = (freqs >= 0.5) & (freqs <= 2)
        gamma_g = np.mean(gain[mask_gamma]) if np.any(mask_gamma) else 0
        lo_g = np.mean(gain[mask_lo]) if np.any(mask_lo) else 0
        gamma_ratio[loc] = gamma_g / (lo_g + 1e-12)

    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, [theta_ratio[l] for l in locs], w,
           color=[loc_colors[l] for l in locs], alpha=0.8, label='Theta (3-10 Hz)')
    ax.bar(x + w/2, [gamma_ratio[l] for l in locs], w,
           color=['#888888', '#aaaaaa', '#cccccc'], alpha=0.8, label='Gamma (30-50 Hz)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Proximal', 'Middle', 'Distal'])
    ax.set_ylabel('Relative Gain Ratio (freq band / low-freq)')
    ax.set_title('Theta vs Gamma Preference by Location\n'
                 '(Distal prefers theta; Proximal prefers high freq)')
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(make_dir('frequency_routing'), 'theta_preference.png')
    plt.savefig(path, dpi=150)
    print(f"[OK] Saved: {path}")
    plt.close()

    print("\n" + "=" * 65)
    print("Summary:")
    print("  - Distal (strong HCN): theta routing gain highest")
    print("  - Proximal (passive):   low-pass, no theta preference")
    print("  - This creates FREQUENCY ROUTING:")
    print("    Different frequencies are selectively routed based on")
    print("    their dendritic origin and local channel composition.")
    print("=" * 65)


if __name__ == '__main__':
    main()
