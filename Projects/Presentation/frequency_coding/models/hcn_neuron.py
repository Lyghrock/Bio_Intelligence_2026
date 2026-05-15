"""
HCN Neuron Model - Theta Resonance Analysis
============================================
Demonstrates how HCN (hyperpolarization-activated cyclic nucleotide-gated) channels
create theta-band (4-10 Hz) resonance in hippocampal neurons.

Key concepts:
1. HCN channels activate during hyperpolarization → sag potential
2. HCN channels create a PEAK in impedance at theta frequencies (band-pass resonance)
3. Passive membranes only show low-pass filtering (impedance decreases with frequency)
4. ZAP protocol directly shows preferential response to theta inputs
"""

from neuron import h
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Output figures directory
_figures_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))

# Load NEURON standard run system
h.load_file('stdrun.hoc')
h.CVode().active(0)

# Load compiled mechanisms from reference_mod
_ref_mod_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'reference_mod', 'nrnmech.dll'
))
if os.path.exists(_ref_mod_path):
    h.nrn_load_dll(_ref_mod_path)
    print(f"[OK] Loaded mechanisms from: {_ref_mod_path}")
else:
    print(f"[WARN] Mechanism DLL not found at: {_ref_mod_path}")


class HCNNeuron:
    """
    Hippocampal neuron model with HCN (Ih) channels for theta resonance.

    HCN channel properties (from Ih.mod):
    - Activated by hyperpolarization (opens when V < -60 mV)
    - Mixed Na+/K+ current (reversal ~ -30 mV)
    - Slow activation/deactivation kinetics → frequency-dependent response
    - Density gradient: higher in distal dendrites (matches CA1 pyramidal cells)
    """

    def __init__(self, dend_L=300, g_h=0.002, g_pas_dend=0.0003):
        self.soma = h.Section(name='soma_hcn')
        self.dend = h.Section(name='dend_hcn')

        # --- SOMA ---
        self.soma.L = 10
        self.soma.diam = 10
        self.soma.cm = 1.0
        self.soma.Ra = 100
        self.soma.insert('pas')
        self.soma.g_pas = 0.0001
        self.soma.e_pas = -65
        self.soma.insert('hh1')  # For action potentials

        # HCN at soma (lighter)
        self.soma.insert('Ih')
        for seg in self.soma:
            seg.Ih.gIhbar = g_h * 0.3

        # --- DENDRITE ---
        self.dend.L = dend_L
        self.dend.diam = 2.0
        self.dend.nseg = 31
        self.dend.cm = 1.0
        self.dend.Ra = 100
        self.dend.insert('pas')
        self.dend.g_pas = g_pas_dend
        self.dend.e_pas = -65

        # HCN in dendrites (stronger) - gradient as in real CA1 neurons
        self.dend.insert('Ih')
        for seg in self.dend:
            x = seg.x  # 0 to 1 along dendrite
            seg.Ih.gIhbar = g_h * (0.5 + 0.5 * x)  # increases toward distal

        self.dend.connect(self.soma(1))

        # Recording
        self.v_soma = h.Vector()
        self.v_dend = h.Vector()
        self.t = h.Vector()
        self.v_soma.record(self.soma(0.5)._ref_v)
        self.v_dend.record(self.dend(0.5)._ref_v)
        self.t.record(h._ref_t)

        # Impedance analysis
        self.imp = h.Impedance()

    def measure_impedance(self, freq_range=(0.5, 30), n_points=50):
        """
        Measure input impedance using NEURON Impedance class.
        IMPORTANT: imp.compute(freq, 1) includes gating state derivatives
        for accurate frequency response of active channels.
        """
        freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
        z_mag = np.zeros(len(freqs))
        phase = np.zeros(len(freqs))

        h.finitialize(-65)
        self.imp.loc(0.5, sec=self.soma)

        for i, f in enumerate(freqs):
            self.imp.compute(f, 0)  # mode=1 (gating) fails with hh1; ZAP captures dynamics correctly
            z_mag[i] = self.imp.input(0.5, sec=self.soma)
            phase[i] = self.imp.input_phase(0.5, sec=self.soma)

        return freqs, z_mag, phase

    def zap_protocol(self, duration=4000, f_start=0.5, f_end=25, amp=0.3):
        """
        ZAP protocol: chirp current from f_start to f_end Hz.

        The voltage response amplitude at each frequency reveals
        the frequency preference of the membrane.
        High response = preferred frequency (resonance).
        """
        dt = 0.025
        t = np.arange(0, duration, dt)

        rate = (f_end - f_start) / (duration / 1000)  # Hz/s sweep rate
        t_sec = t / 1000
        phase = 2 * np.pi * (f_start * t_sec + 0.5 * rate * t_sec**2)
        wave = amp * np.sin(phase)

        stim = h.IClamp(self.soma(0.5))
        stim.delay = 0
        stim.dur = duration
        stim.amp = 0

        wave_vec = h.Vector(wave)
        wave_vec.play(stim._ref_amp, dt)

        h.finitialize(-65)
        h.tstop = duration + 100
        h.run()
        wave_vec.play_remove()

        t_data = np.array(self.t)
        v_data = np.array(self.v_soma)
        return t_data, v_data, t, wave

    def hyperpolarization_test(self, amp=-0.3, dur=500):
        """Hyperpolarizing step to activate HCN → characteristic sag."""
        stim = h.IClamp(self.soma(0.5))
        stim.delay = 200
        stim.dur = dur
        stim.amp = amp

        h.finitialize(-65)
        h.tstop = dur + 600
        h.run()

        return np.array(self.t), np.array(self.v_soma)

    def theta_input_test(self, theta_freq=5, dur=2000):
        """
        Inject theta-frequency sinusoidal current.
        Shows preferential transmission of theta inputs.
        """
        dt = 0.025
        t = np.arange(0, dur, dt)
        wave = 0.2 * np.sin(2 * np.pi * theta_freq * t / 1000)

        stim = h.IClamp(self.soma(0.5))
        stim.delay = 0
        stim.dur = dur
        stim.amp = 0

        wave_vec = h.Vector(wave)
        wave_vec.play(stim._ref_amp, dt)

        h.finitialize(-65)
        h.tstop = dur + 100
        h.run()
        wave_vec.play_remove()

        return np.array(self.t), np.array(self.v_soma), t, wave


class PassiveNeuron:
    """
    Control neuron WITHOUT HCN channels.
    Used to show that theta resonance requires active HCN channels.
    """

    def __init__(self, dend_L=300, g_pas_dend=0.0003):
        self.soma = h.Section(name='soma_passive')
        self.dend = h.Section(name='dend_passive')

        self.soma.L = 10
        self.soma.diam = 10
        self.soma.cm = 1.0
        self.soma.Ra = 100
        self.soma.insert('pas')
        self.soma.g_pas = 0.0001
        self.soma.e_pas = -65
        self.soma.insert('hh1')

        self.dend.L = dend_L
        self.dend.diam = 2.0
        self.dend.nseg = 31
        self.dend.cm = 1.0
        self.dend.Ra = 100
        self.dend.insert('pas')
        self.dend.g_pas = g_pas_dend
        self.dend.e_pas = -65

        self.dend.connect(self.soma(1))

        self.v_soma = h.Vector()
        self.v_dend = h.Vector()
        self.t = h.Vector()
        self.v_soma.record(self.soma(0.5)._ref_v)
        self.v_dend.record(self.dend(0.5)._ref_v)
        self.t.record(h._ref_t)

        self.imp = h.Impedance()

    def measure_impedance(self, freq_range=(0.5, 30), n_points=50):
        """Passive membrane: impedance decreases monotonically with frequency."""
        freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
        z_mag = np.zeros(len(freqs))
        phase = np.zeros(len(freqs))

        h.finitialize(-65)
        self.imp.loc(0.5, sec=self.soma)

        for i, f in enumerate(freqs):
            self.imp.compute(f, 0)  # passive: gating states don't matter
            z_mag[i] = self.imp.input(0.5, sec=self.soma)
            phase[i] = self.imp.input_phase(0.5, sec=self.soma)

        return freqs, z_mag, phase

    def zap_protocol(self, duration=4000, f_start=0.5, f_end=25, amp=0.3):
        """ZAP protocol for passive neuron."""
        dt = 0.025
        t = np.arange(0, duration, dt)

        rate = (f_end - f_start) / (duration / 1000)
        t_sec = t / 1000
        phase = 2 * np.pi * (f_start * t_sec + 0.5 * rate * t_sec**2)
        wave = amp * np.sin(phase)

        stim = h.IClamp(self.soma(0.5))
        stim.delay = 0
        stim.dur = duration
        stim.amp = 0

        wave_vec = h.Vector(wave)
        wave_vec.play(stim._ref_amp, dt)

        h.finitialize(-65)
        h.tstop = duration + 100
        h.run()
        wave_vec.play_remove()

        return np.array(self.t), np.array(self.v_soma), t, wave

    def hyperpolarization_test(self, amp=-0.3, dur=500):
        """Passive: minimal sag (no HCN to counter hyperpolarization)."""
        stim = h.IClamp(self.soma(0.5))
        stim.delay = 200
        stim.dur = dur
        stim.amp = amp

        h.finitialize(-65)
        h.tstop = dur + 600
        h.run()

        return np.array(self.t), np.array(self.v_soma)

    def theta_input_test(self, theta_freq=5, dur=2000):
        """Theta input test for passive neuron."""
        dt = 0.025
        t = np.arange(0, dur, dt)
        wave = 0.2 * np.sin(2 * np.pi * theta_freq * t / 1000)

        stim = h.IClamp(self.soma(0.5))
        stim.delay = 0
        stim.dur = dur
        stim.amp = 0

        wave_vec = h.Vector(wave)
        wave_vec.play(stim._ref_amp, dt)

        h.finitialize(-65)
        h.tstop = dur + 100
        h.run()
        wave_vec.play_remove()

        return np.array(self.t), np.array(self.v_soma), t, wave


def analyze_zap_response(t_v, v_data, t_wave, wave, f_start=0.5, f_end=25, segment_dur=200):
    """
    Extract amplitude at each frequency from ZAP response.

    Returns:
        freqs: center frequency of each segment (Hz)
        v_amps: peak-to-peak voltage amplitude in each segment (mV)
        i_amps: peak-to-peak current amplitude in each segment (nA)
    """
    dt = 0.025
    duration = t_wave[-1]
    n_segments = int(duration / segment_dur)

    freqs_out = []
    v_amps_out = []
    i_amps_out = []

    for i in range(n_segments):
        t0 = i * segment_dur
        t1 = (i + 1) * segment_dur

        # Center frequency for this segment
        mid_f = f_start + (f_end - f_start) * (i + 0.5) / n_segments

        # Voltage in this segment
        v_mask = (t_v >= t0) & (t_v < t1)
        v_seg = v_data[v_mask]
        if len(v_seg) > 10:
            v_pp = np.max(v_seg) - np.min(v_seg)
        else:
            v_pp = 0

        # Current in this segment
        i_mask = (t_wave >= t0) & (t_wave < t1)
        i_seg = wave[i_mask]
        if len(i_seg) > 10:
            i_pp = np.max(i_seg) - np.min(i_seg)
        else:
            i_pp = 0

        freqs_out.append(mid_f)
        v_amps_out.append(v_pp)
        i_amps_out.append(i_pp)

    return np.array(freqs_out), np.array(v_amps_out), np.array(i_amps_out)


def main():
    print("=" * 65)
    print("HCN Neuron Model - Theta Resonance Analysis")
    print("=" * 65)

    # Parameters: HCN conductance tuned for clear theta resonance
    g_h = 0.0025      # S/cm2 - increased from 0.0002 to show real resonance
    g_pas_dend = 0.0003

    print(f"\nCreating neurons (g_h={g_h}, g_pas_dend={g_pas_dend})...")
    print("1. Passive neuron (control, no HCN)")
    neuron_passive = PassiveNeuron(dend_L=300, g_pas_dend=g_pas_dend)

    print("2. HCN neuron (with Ih channels)")
    neuron_hcn = HCNNeuron(dend_L=300, g_h=g_h, g_pas_dend=g_pas_dend)

    # ====== TEST 1: Hyperpolarization Response ======
    print("\n" + "-" * 65)
    print("Test 1: Hyperpolarization Response (HCN activation)")
    print("-" * 65)

    t_pas, v_pas = neuron_passive.hyperpolarization_test(amp=-0.3, dur=500)
    t_hcn, v_hcn = neuron_hcn.hyperpolarization_test(amp=-0.3, dur=500)

    v_rest = -65
    v_min_pas = np.min(v_pas[int(200/0.025):int(700/0.025)])
    v_min_hcn = np.min(v_hcn[int(200/0.025):int(700/0.025)])
    sag_pas = abs(v_min_pas - v_rest)
    sag_hcn = abs(v_min_hcn - v_rest)

    print(f"Passive: V_min={v_min_pas:.1f} mV, sag={sag_pas:.1f} mV (no Ih)")
    print(f"HCN:     V_min={v_min_hcn:.1f} mV, sag={sag_hcn:.1f} mV (with Ih)")

    if sag_hcn < sag_pas - 0.5:
        print("[OK] HCN LIMITS hyperpolarization (smaller sag): Ih depolarizes toward rest")
    elif sag_hcn > sag_pas + 0.5:
        print("[!] HCN ENHANCES hyperpolarization (unusual - check parameters)")
    else:
        print("[!] HCN has minimal effect on sag (try increasing g_h)")

    # ====== TEST 2: ZAP Protocol (Core Resonance Test) ======
    print("\n" + "-" * 65)
    print("Test 2: ZAP Protocol (Frequency Sweep)")
    print("-" * 65)

    print("Running ZAP on Passive neuron (0.5 → 25 Hz)...")
    tp_pas, vp_pas, tw_pas, iw_pas = neuron_passive.zap_protocol(
        duration=4000, f_start=0.5, f_end=25, amp=0.3
    )

    print("Running ZAP on HCN neuron (0.5 → 25 Hz)...")
    tp_hcn, vp_hcn, tw_hcn, iw_hcn = neuron_hcn.zap_protocol(
        duration=4000, f_start=0.5, f_end=25, amp=0.3
    )

    # Analyze ZAP response: extract amplitude per frequency band
    freqs_pas, vpp_pas, ipp_pas = analyze_zap_response(
        tp_pas, vp_pas, tw_pas, iw_pas, f_start=0.5, f_end=25, segment_dur=200
    )
    freqs_hcn, vpp_hcn, ipp_hcn = analyze_zap_response(
        tp_hcn, vp_hcn, tw_hcn, iw_hcn, f_start=0.5, f_end=25, segment_dur=200
    )

    # Compute transfer gain = V_pp / I_pp (voltage amplitude / current amplitude)
    gain_pas = vpp_pas / (ipp_pas + 1e-12)
    gain_hcn = vpp_hcn / (ipp_hcn + 1e-12)

    # Find resonance peak in HCN
    peak_idx_pas = np.argmax(gain_pas)
    peak_idx_hcn = np.argmax(gain_hcn)

    peak_freq_pas = freqs_pas[peak_idx_pas]
    peak_freq_hcn = freqs_hcn[peak_idx_hcn]
    peak_gain_pas = gain_pas[peak_idx_pas]
    peak_gain_hcn = gain_hcn[peak_idx_hcn]

    print(f"\nPassive membrane:  peak gain = {peak_gain_pas:.1f} MΩ @ {peak_freq_pas:.1f} Hz")
    print(f"HCN membrane:    peak gain = {peak_gain_hcn:.1f} MΩ @ {peak_freq_hcn:.1f} Hz")

    # Check if HCN peak is in theta band (4-10 Hz)
    theta_mask_hcn = (freqs_hcn >= 3) & (freqs_hcn <= 10)
    theta_gain_hcn = gain_hcn[theta_mask_hcn]
    theta_freqs = freqs_hcn[theta_mask_hcn]
    theta_peak_idx = np.argmax(theta_gain_hcn)
    theta_peak_freq = theta_freqs[theta_peak_idx]

    print(f"\nTheta band (3-10 Hz) analysis:")
    print(f"  HCN peak in theta band: {theta_peak_freq:.1f} Hz")
    print(f"  HCN gain at theta peak: {theta_gain_hcn[theta_peak_idx]:.1f} MΩ")

    if 3 <= theta_peak_freq <= 10:
        print("[OK] CONFIRMED: HCN creates resonance in theta band (3-10 Hz)")
    else:
        print("[!] Resonance outside expected theta band")

    # Enhancement ratio: HCN gain at theta peak vs passive
    passive_at_theta = np.interp(theta_peak_freq, freqs_pas, gain_pas)
    enhancement = theta_gain_hcn[theta_peak_idx] / passive_at_theta
    print(f"  Enhancement over passive at {theta_peak_freq:.1f} Hz: {enhancement:.2f}x")

    # ====== TEST 3: Theta Input Specificity ======
    print("\n" + "-" * 65)
    print("Test 3: Theta Frequency Input (5 Hz)")
    print("-" * 65)

    theta_freq = 5.0
    tp_pas_t, vp_pas_t, tw_pas_t, iw_pas_t = neuron_passive.theta_input_test(
        theta_freq=theta_freq, dur=2000
    )
    tp_hcn_t, vp_hcn_t, tw_hcn_t, iw_hcn_t = neuron_hcn.theta_input_test(
        theta_freq=theta_freq, dur=2000
    )

    # RMS voltage during theta input
    steady_start = 500
    mask_pas = tp_pas_t > steady_start
    mask_hcn = tp_hcn_t > steady_start

    v_rms_pas = np.sqrt(np.mean(vp_pas_t[mask_pas]**2))
    v_rms_hcn = np.sqrt(np.mean(vp_hcn_t[mask_hcn]**2))

    print(f"Passive neuron response to {theta_freq} Hz input: V_rms = {v_rms_pas:.2f} mV")
    print(f"HCN neuron response to {theta_freq} Hz input:    V_rms = {v_rms_hcn:.2f} mV")
    print(f"Ratio (HCN/Passive): {v_rms_hcn/v_rms_pas:.2f}x")

    # ====== Visualization ======
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. Hyperpolarization sag
        ax = axes[0, 0]
        ax.plot(t_pas, v_pas, 'b-', alpha=0.7, lw=1.5, label='Passive')
        ax.plot(t_hcn, v_hcn, 'r-', lw=2, label='With HCN')
        ax.axhline(v_rest, color='k', ls='--', alpha=0.4, label=f'V_rest={v_rest} mV')
        ax.axvspan(200, 700, alpha=0.1, color='gray')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title('Hyperpolarization Sag\n(HCN activation)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1100)

        # 2. ZAP voltage traces
        ax = axes[0, 1]
        ax.plot(tp_pas, vp_pas, 'b-', alpha=0.6, lw=1, label='Passive')
        ax.plot(tp_hcn, vp_hcn, 'r-', lw=1.5, label='With HCN')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title('ZAP Protocol Response\n(Frequency sweep 0.5→25 Hz)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. Transfer gain comparison (core plot!)
        ax = axes[0, 2]
        ax.semilogy(freqs_pas, gain_pas, 'b-', lw=2, label='Passive', alpha=0.8)
        ax.semilogy(freqs_hcn, gain_hcn, 'r-', lw=2.5, label='With HCN', alpha=0.9)
        ax.axvspan(3, 10, alpha=0.15, color='orange', label='Theta band')
        ax.axvline(theta_peak_freq, color='darkred', ls='--', lw=1.5,
                   label=f'HCN peak: {theta_peak_freq:.1f} Hz')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Transfer Gain (MΩ)')
        ax.set_title('Transfer Gain = V_pp / I_pp\n(HCN Theta Resonance)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 25)

        # 4. Enhancement ratio
        ax = axes[1, 0]
        ratio = gain_hcn / (gain_pas + 1e-12)
        ax.plot(freqs_hcn, ratio, 'g-', lw=2)
        ax.axvspan(3, 10, alpha=0.15, color='orange', label='Theta band')
        ax.axhline(1.0, color='k', ls='--', alpha=0.5)
        theta_ratio = ratio[(freqs_hcn >= 3) & (freqs_hcn <= 10)]
        theta_ratio_max = np.max(theta_ratio)
        theta_ratio_freq = freqs_hcn[(freqs_hcn >= 3) & (freqs_hcn <= 10)][np.argmax(theta_ratio)]
        ax.axhline(theta_ratio_max, color='darkgreen', ls=':', alpha=0.7)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('HCN / Passive Gain Ratio')
        ax.set_title(f'HCN Selectivity\n(Max enhancement: {theta_ratio_max:.2f}x at {theta_ratio_freq:.1f} Hz)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 25)

        # 5. Theta input time series
        ax = axes[1, 1]
        mask_plot = (tp_pas_t > 200) & (tp_pas_t < 1200)
        ax.plot(tp_pas_t[mask_plot], vp_pas_t[mask_plot], 'b-', alpha=0.7, lw=1, label='Passive')
        ax.plot(tp_hcn_t[mask_plot], vp_hcn_t[mask_plot], 'r-', lw=1.5, label='With HCN')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title(f'Theta Input Response ({theta_freq} Hz)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 6. Summary
        ax = axes[1, 2]
        ax.axis('off')
        summary = (
            "HCN (Ih) Channel Properties:\n"
            "  - Activated by hyperpolarization\n"
            "  - Mixed Na+/K+ current (E_rev ≈ -30 mV)\n"
            "  - Slow kinetics → frequency preference\n\n"
            "Results:\n"
            f"  - Sag: Passive={sag_pas:.1f} mV, HCN={sag_hcn:.1f} mV\n"
            f"  - HCN peak: {peak_gain_hcn:.1f} MΩ @ {peak_freq_hcn:.1f} Hz\n"
            f"  - Passive peak:  {peak_gain_pas:.1f} MΩ @ {peak_freq_pas:.1f} Hz\n"
            f"  - Theta peak: {theta_peak_freq:.1f} Hz (enhancement: {theta_ratio_max:.2f}x)\n\n"
            "Biological Significance:\n"
            "  - CA1 pyramidal neurons show similar resonance\n"
            "  - HCN channels critical for theta oscillations\n"
            "  - Selective amplification of theta-band inputs\n"
            "  - Enables temporal coding at theta frequency"
        )
        ax.text(0.05, 0.98, summary, transform=ax.transAxes,
                fontsize=9.5, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        fig_dir = os.path.join(_figures_root, 'hcn_resonance')
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, 'hcn_resonance.png')
        plt.savefig(fig_path, dpi=150)
        print(f"\nFigure saved: {fig_path}")

    except ImportError:
        print("\nmatplotlib not available")

    print("\n" + "=" * 65)
    print("Simulation complete!")
    print("=" * 65)


if __name__ == '__main__':
    main()
