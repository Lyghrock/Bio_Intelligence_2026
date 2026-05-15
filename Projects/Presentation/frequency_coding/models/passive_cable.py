"""
Passive Cable Model - Part 1: Passive Cable as Low-Pass Filter
================================================================

Sub-experiments (each saves to its own figure file):
  Test 1: Z_impedance      -> figures/passive_impedance/z_impedance.png
  Test 2: Z_transfer        -> figures/passive_impedance/z_transfer.png
  Test 3: Signal filtering -> figures/signal_filtering/snr_improvement.png
  Test 4: Freq attenuation -> figures/signal_filtering/freq_attenuation.png
"""

from neuron import h
import numpy as np
import os
from scipy.signal import butter, sosfilt, welch

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


class PassiveCable:
    """
    Passive cable: single soma + long dendrite, passive membrane only.
    Serves as the baseline filter against which active channels are compared.
    """

    def __init__(self, L_dend=500, nseg=50, d=2.0):
        self.soma = h.Section(name='soma')
        self.soma.L = 10
        self.soma.diam = 10
        self.soma.cm = 1.0
        self.soma.Ra = 100
        self.soma.insert('pas')
        self.soma.g_pas = 0.0003
        self.soma.e_pas = -65

        self.dend = h.Section(name='dend')
        self.dend.L = L_dend
        self.dend.diam = d
        self.dend.nseg = nseg
        self.dend.cm = 1.0
        self.dend.Ra = 100
        self.dend.insert('pas')
        self.dend.g_pas = 0.0003
        self.dend.e_pas = -65
        self.dend.connect(self.soma(1))

        self.L_dend = L_dend
        self.d = d
        self.lambda_um = np.sqrt(d * 1e-4 / (4 * np.pi * 100 * 0.0003)) * 1e4

        self.v_soma = h.Vector()
        self.v_dend = h.Vector()
        self.t = h.Vector()
        self.v_soma.record(self.soma(0.5)._ref_v)
        self.v_dend.record(self.dend(0.9)._ref_v)
        self.t.record(h._ref_t)

        self.imp = h.Impedance()

    # ------------------------------------------------------------------
    # Test 1: Input Impedance at Soma vs Dendrite
    # ------------------------------------------------------------------
    def measure_input_impedance(self, freq_range=(0.1, 100), n_points=30):
        """
        Impedance at soma (0) and distal dendrite (0.9) vs frequency.
        Expect: Z decreases with frequency (low-pass behavior).
        """
        freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
        z_soma = np.zeros(len(freqs))
        z_dend = np.zeros(len(freqs))

        h.finitialize(-65)

        for i, f in enumerate(freqs):
            # Soma
            self.imp.loc(0.5, sec=self.soma)
            self.imp.compute(f, 0)
            z_soma[i] = self.imp.input(0.5, sec=self.soma)

            # Distal dendrite
            self.imp.loc(0.9, sec=self.dend)
            self.imp.compute(f, 0)
            z_dend[i] = self.imp.input(0.9, sec=self.dend)

        return freqs, z_soma, z_dend

    def plot_z_impedance(self, freqs, z_soma, z_dend):
        """
        Save: figures/passive_impedance/z_impedance.png
        Shows how passive membrane impedance drops with frequency.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: absolute impedance
        ax = axes[0]
        ax.semilogx(freqs, z_soma, 'b-o', ms=4, lw=1.5, label='Soma (0)')
        ax.semilogx(freqs, z_dend, 'r-s', ms=4, lw=1.5, label='Distal dend (0.9)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Input Impedance (MΩ)')
        ax.set_title('Passive Cable: Input Impedance vs Frequency\n(Low-pass behavior)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate key values
        for f_target, label in [(1, '1 Hz'), (50, '50 Hz')]:
            idx = np.argmin(np.abs(freqs - f_target))
            ax.annotate(f'{z_soma[idx]:.0f} MΩ',
                        xy=(freqs[idx], z_soma[idx]),
                        xytext=(5, -10), textcoords='offset points', fontsize=8)

        # Right: ratio
        ax = axes[1]
        ratio = z_dend / (z_soma + 1e-12)
        ax.semilogx(freqs, ratio, 'g-', lw=2)
        ax.axhline(1.0, color='k', ls='--', alpha=0.4)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Z_dend / Z_soma')
        ax.set_title('Spatial Impedance Ratio\n(Distal always lower than soma)')
        ax.grid(True, alpha=0.3)

        fig.text(0.5, 0.01,
                 f'L_dend={self.L_dend} μm, d={self.d} μm, λ={self.lambda_um:.1f} μm',
                 ha='center', fontsize=9, color='gray')

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig_dir = os.path.join(_figures_root, 'passive_impedance')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, 'z_impedance.png')
        plt.savefig(path, dpi=150)
        print(f"[OK] Saved: {path}")
        plt.close()

    # ------------------------------------------------------------------
    # Test 2: Transfer Impedance (Dendrite -> Soma)
    # ------------------------------------------------------------------
    def inject_sinusoidal(self, freq, amp=0.1, duration=1500):
        """Inject sine wave at distal dendrite, return (t, v_soma, v_dend)."""
        dt = 0.025
        t_vec = np.arange(0, duration, dt)
        wave = amp * np.sin(2 * np.pi * freq * t_vec / 1000)

        v_soma_r = h.Vector()
        v_dend_r = h.Vector()
        t_r = h.Vector()
        v_soma_r.record(self.soma(0.5)._ref_v)
        v_dend_r.record(self.dend(0.9)._ref_v)
        t_r.record(h._ref_t)

        stim = h.IClamp(self.dend(0.9))
        stim.delay = 0
        stim.dur = duration
        stim.amp = 0
        h.Vector(wave).play(stim._ref_amp, dt)

        h.dt = dt
        h.tstop = duration
        h.finitialize(-65)
        h.run()
        h.Vector(wave).play_remove()

        return np.array(t_r), np.array(v_soma_r), np.array(v_dend_r)

    def measure_transfer_impedance(self, freq_range=(0.1, 100), n_points=30):
        """
        Transfer impedance: voltage at soma caused by current at distal dendrite.
        Lower at high frequencies = high-freq signals attenuate more.
        """
        freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
        z_transfer = np.zeros(len(freqs))
        phase = np.zeros(len(freqs))

        h.finitialize(-65)

        for i, f in enumerate(freqs):
            self.imp.loc(0.9, sec=self.dend)
            self.imp.compute(f, 0)
            z_transfer[i] = self.imp.transfer(0.5, sec=self.soma)
            phase[i] = self.imp.transfer_phase(0.5, sec=self.soma)

        return freqs, z_transfer, phase

    def plot_z_transfer(self, freqs, z_transfer, phase):
        """
        Save: figures/passive_impedance/z_transfer.png
        Shows how signals from distal dendrite attenuate before reaching soma.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.semilogx(freqs, z_transfer, 'b-o', ms=4, lw=1.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Transfer Impedance (MΩ)')
        ax.set_title('Transfer Impedance: Distal Dendrite → Soma\n(Low frequencies pass; high frequencies attenuate)')
        ax.grid(True, alpha=0.3)
        # Key values
        for f_target in [1, 10, 50]:
            idx = np.argmin(np.abs(freqs - f_target))
            ax.annotate(f'{z_transfer[idx]:.1f} MΩ',
                        xy=(freqs[idx], z_transfer[idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax = axes[1]
        ax.semilogx(freqs, np.degrees(phase), 'r-', lw=2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase (degrees)')
        ax.set_title('Transfer Phase\n(Lag increases with frequency)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_dir = os.path.join(_figures_root, 'passive_impedance')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, 'z_transfer.png')
        plt.savefig(path, dpi=150)
        print(f"[OK] Saved: {path}")
        plt.close()

    # ------------------------------------------------------------------
    # Test 3: Signal Filtering (SNR Improvement)
    # ------------------------------------------------------------------
    def inject_noisy_signal(self, duration=3000, signal_freq=5, snr_db=-3.0):
        """
        Inject: 5 Hz sine + band-passed noise (30-100 Hz).
        Returns dict with all time-series.
        """
        dt = 0.025
        t_vec = np.arange(0, duration, dt)

        signal = 0.3 * np.sin(2 * np.pi * signal_freq * t_vec / 1000)
        noise_amp = 0.3 / (10 ** (snr_db / 20.0))

        np.random.seed(42)
        noise = noise_amp * np.random.randn(len(t_vec))
        sos = butter(2, [30, 100], btype='band', fs=40000, output='sos')
        high_freq_noise = sosfilt(sos, noise) * 0.5

        input_current = signal + high_freq_noise

        v_soma_r = h.Vector()
        v_dend_r = h.Vector()
        t_r = h.Vector()
        v_soma_r.record(self.soma(0.5)._ref_v)
        v_dend_r.record(self.dend(0.9)._ref_v)
        t_r.record(h._ref_t)

        stim = h.IClamp(self.dend(0.9))
        stim.delay = 0
        stim.dur = duration
        stim.amp = 0
        h.Vector(input_current).play(stim._ref_amp, dt)

        h.dt = dt
        h.tstop = duration
        h.finitialize(-65)
        h.run()
        h.Vector(input_current).play_remove()

        return {
            't': np.array(t_r),
            'input_current': input_current,
            'v_soma': np.array(v_soma_r),
            'v_dend': np.array(v_dend_r),
            'signal_freq': signal_freq,
            'snr_db': snr_db
        }

    def plot_snr_improvement(self, result):
        """
        Save: figures/signal_filtering/snr_improvement.png
        Shows input vs output in time domain + PSD comparison.
        """
        t = result['t']
        inp = result['input_current']
        v_soma = result['v_soma']
        signal_freq = result['signal_freq']
        snr_db = result['snr_db']

        # Skip transient
        skip = int(500 / 0.025)
        t_ss = t[skip:]
        inp_ss = inp[skip:]
        v_ss = v_soma[skip:]
        min_len = min(len(t_ss), len(inp_ss), len(v_ss))
        t_ss = t_ss[:min_len]
        inp_ss = inp_ss[:min_len]
        v_ss = v_ss[:min_len]

        # PSD
        freqs_w, psd_inp = welch(inp_ss, fs=40000, nperseg=min(131072, len(inp_ss)), noverlap=65536)
        freqs_w, psd_soma = welch(v_ss, fs=40000, nperseg=min(131072, len(v_ss)), noverlap=65536)

        # SNR bands
        sig_mask = (freqs_w >= signal_freq - 3) & (freqs_w <= signal_freq + 3) & (freqs_w > 1)
        noise_mask = (freqs_w >= 30) & (freqs_w <= 100)

        inp_sig = np.mean(psd_inp[sig_mask]) if np.any(sig_mask) else 0
        inp_noise = np.mean(psd_inp[noise_mask]) if np.any(noise_mask) else 0
        out_sig = np.mean(psd_soma[sig_mask]) if np.any(sig_mask) else 0
        out_noise = np.mean(psd_soma[noise_mask]) if np.any(noise_mask) else 0

        def _snr(s, n, floor=1e-30):
            """SNR in dB, with noise floor to avoid -inf."""
            n_safe = max(n, floor)
            if s <= 0:
                return -np.inf
            return 10 * np.log10(s / n_safe)

        snr_in = _snr(inp_sig, inp_noise)
        snr_out = _snr(out_sig, out_noise)
        snr_imp = snr_out - snr_in

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # 1. Time series - input
        ax = axes[0, 0]
        n_show = 2000
        ax.plot(t_ss[:n_show] - t_ss[0], inp_ss[:n_show] * 1e3, 'b-', alpha=0.8, lw=0.8)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Current (pA)')
        ax.set_title(f'Input: {signal_freq} Hz signal + 30-100 Hz noise  (SNR={result["snr_db"]:.1f} dB)')
        ax.grid(True, alpha=0.3)

        # 2. Time series - soma
        ax = axes[0, 1]
        ax.plot(t_ss[:n_show] - t_ss[0], v_ss[:n_show], 'r-', lw=1.0)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title(f'Soma output: noise filtered  (SNR={snr_out:.1f} dB)')
        ax.grid(True, alpha=0.3)

        # 3. PSD comparison
        ax = axes[1, 0]
        ax.semilogy(freqs_w, psd_inp * 1e6, 'b-', alpha=0.5, lw=0.8, label='Input current')
        ax.semilogy(freqs_w, psd_soma, 'r-', lw=2, label='Soma voltage')
        ax.axvline(signal_freq, color='g', ls='--', alpha=0.8, label=f'{signal_freq} Hz signal')
        ax.axvspan(30, 100, alpha=0.15, color='gray', label='Noise band')
        ax.set_xlim(0, 120)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (log)')
        ax.set_title('Frequency Spectrum: Input vs Soma')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Summary
        ax = axes[1, 1]
        ax.axis('off')
        txt = (
            f"Signal: {signal_freq} Hz sine wave\n"
            f"Noise: 30-100 Hz band-passed noise\n"
            f"Input SNR:  {snr_in:.1f} dB\n"
            f"Output SNR: {snr_out:.1f} dB\n"
            f"SNR Improvement: {snr_imp:.1f} dB\n\n"
            f"Result:\n"
            f"  Passive cable acts as LOW-PASS FILTER\n"
            f"  High-freq noise shunted through C_m\n"
            f"  Low-freq signal passes through R_m\n"
            f"  -> Natural denoising at no metabolic cost\n\n"
            f"Mechanism:\n"
            f"  Z_c = R_m / sqrt(1 + (2πf·R_m·C_m)²)\n"
            f"  At high f: Z_c → 1/(2πf·C_m)"
        )
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=10, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

        plt.tight_layout()
        fig_dir = os.path.join(_figures_root, 'signal_filtering')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, 'snr_improvement.png')
        plt.savefig(path, dpi=150)
        print(f"[OK] Saved: {path}  (SNR improvement: {snr_imp:.1f} dB)")
        plt.close()

        return {'snr_in': snr_in, 'snr_out': snr_out, 'snr_imp': snr_imp}

    # ------------------------------------------------------------------
    # Test 4: Frequency-Dependent Attenuation
    # ------------------------------------------------------------------
    def inject_single_frequency(self, freq, amp=0.1, duration=1500):
        """Inject pure sine wave at given frequency, return soma V amplitude."""
        dt = 0.025
        t_vec = np.arange(0, duration, dt)
        wave = amp * np.sin(2 * np.pi * freq * t_vec / 1000)

        v_soma_r = h.Vector()
        t_r = h.Vector()
        v_soma_r.record(self.soma(0.5)._ref_v)
        t_r.record(h._ref_t)

        stim = h.IClamp(self.dend(0.9))
        stim.delay = 0
        stim.dur = duration
        stim.amp = 0
        h.Vector(wave).play(stim._ref_amp, dt)

        h.dt = dt
        h.tstop = duration
        h.finitialize(-65)
        h.run()
        h.Vector(wave).play_remove()

        v = np.array(v_soma_r)
        skip = int(200 / 0.025)
        v_ss = v[skip:]
        v_amp = (np.max(v_ss) - np.min(v_ss)) / 2
        return v_amp

    def plot_freq_attenuation(self):
        """
        Save: figures/signal_filtering/freq_attenuation.png
        Directly measures how amplitude of soma response drops with frequency.
        """
        test_freqs = [0.5, 1, 2, 5, 10, 20, 40, 60, 80]
        amps = []
        for f in test_freqs:
            a = self.inject_single_frequency(f, amp=0.1, duration=1500)
            amps.append(a)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.semilogx(test_freqs, amps, 'bo-', ms=6, lw=2)
        ax.set_xlabel('Input Frequency (Hz)')
        ax.set_ylabel('Soma Response Amplitude (mV)')
        ax.set_title('Frequency-Dependent Attenuation\n(Low-pass: amplitude drops at high frequency)')
        ax.grid(True, alpha=0.3)
        # Annotate
        for f, a in [(1, amps[1]), (10, amps[4]), (80, amps[8])]:
            idx = test_freqs.index(f)
            ax.annotate(f'{a:.1f} mV', xy=(f, a),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        # Normalized
        ax = axes[1]
        amps_norm = np.array(amps) / (amps[0] + 1e-12)
        ax.semilogx(test_freqs, amps_norm, 'ro-', ms=6, lw=2)
        ax.axhline(0.5, color='k', ls='--', alpha=0.5, label='50% attenuation')
        ax.set_xlabel('Input Frequency (Hz)')
        ax.set_ylabel('Normalized Amplitude (rel to 0.5 Hz)')
        ax.set_title('Normalized Attenuation Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cutoff frequency (where amplitude drops to 1/sqrt(2))
        cutoff_idx = np.where(amps_norm < 0.707)[0]
        if len(cutoff_idx) > 0:
            f_cutoff = test_freqs[cutoff_idx[0]]
            ax.axvline(f_cutoff, color='darkred', ls=':', alpha=0.7,
                       label=f'-3dB cutoff ≈ {f_cutoff} Hz')
            ax.legend()

        fig.text(0.5, 0.01,
                 f'Dendrite length={self.L_dend} μm, diam={self.d} μm, λ={self.lambda_um:.1f} μm',
                 ha='center', fontsize=9, color='gray')

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig_dir = os.path.join(_figures_root, 'signal_filtering')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, 'freq_attenuation.png')
        plt.savefig(path, dpi=150)
        print(f"[OK] Saved: {path}")
        plt.close()


# ==================================================================
# Main: run all 4 sub-experiments
# ==================================================================
def main():
    print("=" * 65)
    print("PASSIVE CABLE: Low-Pass Filtering")
    print("=" * 65)

    model = PassiveCable(L_dend=500, nseg=50, d=2.0)
    print(f"\nModel: L={model.L_dend} μm, d={model.d} μm, λ={model.lambda_um:.1f} μm")

    # ---- Test 1: Input Impedance ----
    print("\n" + "-" * 65)
    print("Test 1: Input Impedance (Soma vs Distal Dendrite)")
    print("-" * 65)
    freqs, z_soma, z_dend = model.measure_input_impedance(freq_range=(0.1, 100))
    print(f"  Z_soma @ 1 Hz:  {z_soma[np.argmin(np.abs(freqs-1))]:.1f} MΩ")
    print(f"  Z_soma @ 50 Hz: {z_soma[np.argmin(np.abs(freqs-50))]:.1f} MΩ")
    ratio_1 = z_soma[np.argmin(np.abs(freqs-1))] / z_soma[np.argmin(np.abs(freqs-50))]
    print(f"  Ratio (1Hz/50Hz): {ratio_1:.2f}x")
    if ratio_1 > 1:
        print("  [OK] CONFIRMED: Impedance decreases with frequency (low-pass)")
    model.plot_z_impedance(freqs, z_soma, z_dend)

    # ---- Test 2: Transfer Impedance ----
    print("\n" + "-" * 65)
    print("Test 2: Transfer Impedance (Distal Dendrite -> Soma)")
    print("-" * 65)
    freqs_t, z_tr, phase_tr = model.measure_transfer_impedance(freq_range=(0.1, 100))
    print(f"  Z_transfer @ 1 Hz:  {z_tr[np.argmin(np.abs(freqs_t-1))]:.1f} MΩ")
    print(f"  Z_transfer @ 50 Hz: {z_tr[np.argmin(np.abs(freqs_t-50))]:.1f} MΩ")
    ratio_t = z_tr[np.argmin(np.abs(freqs_t-1))] / z_tr[np.argmin(np.abs(freqs_t-50))]
    print(f"  Ratio (1Hz/50Hz): {ratio_t:.2f}x")
    if ratio_t > 1:
        print("  [OK] CONFIRMED: High-frequency signals attenuate more")
    model.plot_z_transfer(freqs_t, z_tr, phase_tr)

    # ---- Test 3: SNR Improvement ----
    print("\n" + "-" * 65)
    print("Test 3: Signal Filtering (SNR Improvement)")
    print("-" * 65)
    result = model.inject_noisy_signal(duration=3000, signal_freq=5, snr_db=-3.0)
    snr_data = model.plot_snr_improvement(result)
    if snr_data['snr_imp'] > 0:
        print(f"  [OK] CONFIRMED: SNR improved by {snr_data['snr_imp']:.1f} dB")
    else:
        print(f"  [!] SNR change: {snr_data['snr_imp']:.1f} dB")

    # ---- Test 4: Frequency Attenuation ----
    print("\n" + "-" * 65)
    print("Test 4: Frequency-Dependent Attenuation")
    print("-" * 65)
    print("  Measuring response at: 0.5, 1, 2, 5, 10, 20, 40, 60, 80 Hz...")
    model.plot_freq_attenuation()

    print("\n" + "=" * 65)
    print("PASSIVE CABLE experiments complete!")
    print("=" * 65)
    print("\nSummary:")
    print("  1. Low-pass: impedance drops from ~{:.0f}MΩ (1Hz) to ~{:.0f}MΩ (50Hz)".format(
          z_soma[np.argmin(np.abs(freqs-1))], z_soma[np.argmin(np.abs(freqs-50))]))
    print("  2. Signal attenuation: high-freq signals attenuated ~{:.1f}x more".format(ratio_t))
    print("  3. SNR improvement: +{:.1f} dB (membrane capacitance filters noise)".format(
          snr_data['snr_imp']))
    print("\nConclusion: Passive cable acts as natural low-pass filter.")


if __name__ == '__main__':
    main()
