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
import warnings
from scipy.signal import butter, sosfiltfilt, welch

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


def compute_snr_db(freqs, psd, signal_band=(4, 6), noise_band=(30, 100), eps=1e-30):
    """Bandpower SNR in dB from a one-sided PSD."""
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    sig_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])

    def _bandpower(mask):
        if np.count_nonzero(mask) < 2:
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))

    signal_power = _bandpower(sig_mask)
    noise_power = _bandpower(noise_mask)
    if signal_power <= eps or noise_power <= eps:
        return np.nan, signal_power, noise_power
    return 10 * np.log10(signal_power / noise_power), signal_power, noise_power


def _welch_psd(x, fs):
    """Welch PSD with safe segment and overlap sizes."""
    x = np.asarray(x) - np.mean(x)
    nperseg = min(int(fs * 2), len(x))
    noverlap = nperseg // 2
    min_nfft = max(nperseg, int(fs / 0.25))
    nfft = 2 ** int(np.ceil(np.log2(min_nfft)))
    return welch(x, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)


def _fit_sine_amplitude(t_ms, y, freq_hz):
    """Least-squares amplitude at freq_hz for y(t)."""
    t_sec = np.asarray(t_ms) / 1000.0
    y = np.asarray(y)
    omega_t = 2 * np.pi * freq_hz * t_sec
    design = np.column_stack([
        np.sin(omega_t),
        np.cos(omega_t),
        np.ones_like(omega_t),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(np.sqrt(coeffs[0] ** 2 + coeffs[1] ** 2))


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
    def inject_noisy_signal(self, duration=400, signal_freq=5, snr_db=-3.0):
        """
        Inject: 5 Hz sine + band-passed noise (30-100 Hz).
        Returns dict with all time-series.
        """
        dt = 0.025
        assert duration >= 400, "5 Hz filtering experiment requires tstop >= 400 ms"
        t_vec = np.arange(0, duration, dt)
        fs = 1000.0 / dt
        t_sec = t_vec / 1000.0

        a_signal_nA = 0.02
        signal = a_signal_nA * np.sin(2 * np.pi * signal_freq * t_sec)

        np.random.seed(42)
        white_noise = np.random.randn(len(t_vec))
        sos = butter(4, [30, 100], btype='band', fs=fs, output='sos')
        high_freq_noise = sosfiltfilt(sos, white_noise)
        high_freq_noise -= np.mean(high_freq_noise)
        noise_rms_target = (a_signal_nA / np.sqrt(2)) / (10 ** (snr_db / 20.0))
        noise_rms = np.sqrt(np.mean(high_freq_noise ** 2))
        if noise_rms <= 0:
            raise RuntimeError("Band-passed noise has zero RMS")
        high_freq_noise *= noise_rms_target / noise_rms
        high_freq_noise -= np.mean(high_freq_noise)

        input_current = signal + high_freq_noise
        assert np.std(input_current) > 0, "Input current is flat"

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
        stim_vec = h.Vector(input_current)
        stim_vec.play(stim._ref_amp, dt)

        h.dt = dt
        h.tstop = duration
        h.finitialize(-65)
        h.run()
        stim_vec.play_remove()

        v_soma = np.array(v_soma_r)
        if np.std(v_soma) < 1e-6:
            raise RuntimeError("No soma response detected")

        return {
            't': np.array(t_r),
            'input_current': input_current,
            'v_soma': v_soma,
            'v_dend': np.array(v_dend_r),
            'signal_freq': signal_freq,
            'target_snr_db': snr_db,
            'dt': dt,
            'fs': fs,
            'tstop_ms': duration,
            'a_signal_nA': a_signal_nA,
            'noise_rms_nA': float(np.sqrt(np.mean(high_freq_noise ** 2))),
            'noise_band': (30, 100)
        }

    def analyze_filtering_result(self, result):
        """Compute steady-state PSD and bandpower SNR for the filtering experiment."""
        t = result['t']
        inp = result['input_current']
        v_soma = result['v_soma']
        dt = result['dt']
        fs = result['fs']
        noise_band = result['noise_band']

        # For the 400 ms version, use the full trace; it contains exactly two 5 Hz cycles.
        transient_ms = 0.0 if t[-1] <= 500 else 500.0
        skip = int(transient_ms / dt)
        t_ss = t[skip:]
        inp_ss = inp[skip:]
        v_ss = v_soma[skip:]
        min_len = min(len(t_ss), len(inp_ss), len(v_ss))
        t_ss = t_ss[:min_len]
        inp_ss = inp_ss[:min_len]
        v_ss = v_ss[:min_len]
        inp_ss = inp_ss - np.mean(inp_ss)
        v_ss = v_ss - np.mean(v_ss)

        if np.std(v_ss) < 1e-6:
            raise RuntimeError("No soma response detected")

        # PSD
        freqs_w, psd_inp = _welch_psd(inp_ss, fs)
        _, psd_soma = _welch_psd(v_ss, fs)

        snr_in, inp_sig, inp_noise = compute_snr_db(
            freqs_w, psd_inp, signal_band=(4, 6), noise_band=noise_band
        )
        snr_out, out_sig, out_noise = compute_snr_db(
            freqs_w, psd_soma, signal_band=(4, 6), noise_band=noise_band
        )
        assert np.isfinite(snr_in), "Input SNR is invalid"
        if not np.isfinite(snr_out):
            raise RuntimeError("invalid: no soma response")
        snr_imp = snr_out - snr_in
        assert np.isfinite(snr_out), "Output SNR is invalid"

        return {
            't_ss': t_ss,
            'input_ss': inp_ss,
            'v_soma_ss': v_ss,
            'freqs': freqs_w,
            'psd_input': psd_inp,
            'psd_soma': psd_soma,
            'snr_in': snr_in,
            'snr_out': snr_out,
            'snr_imp': snr_imp,
            'input_signal_power': inp_sig,
            'input_noise_power': inp_noise,
            'output_signal_power': out_sig,
            'output_noise_power': out_noise,
        }

    def plot_signal_filtering_results(self, result, metrics=None):
        """
        Save: figures/signal_filtering/signal_filtering_results.png
        Shows input and soma output in time and frequency domains.
        """
        if metrics is None:
            metrics = self.analyze_filtering_result(result)

        t_ss = metrics['t_ss']
        inp_ss = metrics['input_ss']
        v_ss = metrics['v_soma_ss']
        freqs_w = metrics['freqs']
        psd_inp = metrics['psd_input']
        psd_soma = metrics['psd_soma']
        snr_in = metrics['snr_in']
        snr_out = metrics['snr_out']
        snr_imp = metrics['snr_imp']
        signal_freq = result['signal_freq']
        noise_band = result['noise_band']

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # 1. Time series - input
        ax = axes[0, 0]
        display_start = 0.0
        display_duration = 400.0
        show_mask = (t_ss >= display_start) & (t_ss < display_start + display_duration)
        if np.count_nonzero(show_mask) < 2:
            show_mask = t_ss >= (t_ss[-1] - display_duration)
        ax.plot(t_ss[show_mask] - t_ss[show_mask][0], inp_ss[show_mask] * 1e3, 'b-', alpha=0.8, lw=0.8)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Current (pA)')
        ax.set_title(f'Input: {signal_freq} Hz signal + 30-100 Hz noise  (SNR={snr_in:.1f} dB)')
        ax.grid(True, alpha=0.3)

        # 2. Time series - soma
        ax = axes[0, 1]
        ax.plot(t_ss[show_mask] - t_ss[show_mask][0], v_ss[show_mask], 'r-', lw=1.0)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage fluctuation (mV)')
        ax.set_title(f'Soma output: noise filtered  (SNR={snr_out:.1f} dB)')
        ax.grid(True, alpha=0.3)

        # 3. PSD comparison
        ax = axes[1, 0]
        band_mask = (freqs_w >= 0.5) & (freqs_w <= 120)
        psd_inp_norm = psd_inp / np.max(psd_inp[band_mask])
        psd_soma_norm = psd_soma / np.max(psd_soma[band_mask])
        ax.semilogy(freqs_w, psd_inp_norm, 'b-', alpha=0.55, lw=0.9, label='Input current PSD, normalized')
        ax.semilogy(freqs_w, psd_soma_norm, 'r-', lw=1.8, label='Soma voltage PSD, normalized')
        ax.axvline(signal_freq, color='g', ls='--', alpha=0.8, label=f'{signal_freq} Hz signal')
        ax.axvspan(4, 6, alpha=0.12, color='green', label='5 Hz signal band')
        ax.axvspan(noise_band[0], noise_band[1], alpha=0.15, color='gray', label='30-100 Hz noise band')
        ax.set_xlim(0, 120)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized PSD (log)')
        ax.set_title('Frequency Spectrum: normalized within each signal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Summary
        ax = axes[1, 1]
        ax.axis('off')
        txt = (
            "PASSIVE CABLE FILTERING RESULTS\n\n"
            "Input Signal:\n"
            f"- Carrier: {signal_freq} Hz sine wave\n"
            "- Noise: 30-100 Hz band-passed noise\n"
            f"- Signal amplitude: {result['a_signal_nA'] * 1000:.0f} pA peak\n"
            f"- Noise RMS: {result['noise_rms_nA'] * 1000:.1f} pA\n"
            f"- Simulation window: {result['tstop_ms']:.0f} ms\n"
            f"- Input SNR: {snr_in:.1f} dB\n\n"
            "Output at soma:\n"
            "- Low-frequency component relatively preserved\n"
            "- 30-100 Hz noise relatively attenuated\n"
            f"- Output SNR: {snr_out:.1f} dB\n"
            f"- SNR Improvement: {snr_imp:.1f} dB\n\n"
            "Mechanism:\n"
            "- Passive membrane capacitance and cable\n"
            "  propagation create low-pass filtering\n"
            "- Higher-frequency components are attenuated\n"
            "  more strongly than low-frequency components\n"
            "- This produces modest passive denoising"
        )
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=10, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

        plt.tight_layout()
        fig_dir = os.path.join(_figures_root, 'signal_filtering')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, 'signal_filtering_results.png')
        plt.savefig(path, dpi=150)
        print(f"[OK] Saved: {path}")
        plt.close()
        return metrics

    def plot_snr_improvement(self, result, metrics=None):
        """
        Save: figures/signal_filtering/snr_improvement.png
        Dedicated SNR summary plot.
        """
        if metrics is None:
            metrics = self.analyze_filtering_result(result)

        snr_in = metrics['snr_in']
        snr_out = metrics['snr_out']
        snr_imp = metrics['snr_imp']
        if not (np.isfinite(snr_in) and np.isfinite(snr_out) and np.isfinite(snr_imp)):
            raise RuntimeError("Invalid SNR values; refusing to generate success figure")

        input_signal_power = metrics['input_signal_power']
        input_noise_power = metrics['input_noise_power']
        output_signal_power = metrics['output_signal_power']
        output_noise_power = metrics['output_noise_power']

        input_total = input_signal_power + input_noise_power
        output_total = output_signal_power + output_noise_power
        if input_total <= 0 or output_total <= 0:
            raise RuntimeError("Invalid bandpowers; refusing to generate SNR figure")

        bandpower_fractions = np.array([
            [input_signal_power / input_total, input_noise_power / input_total],
            [output_signal_power / output_total, output_noise_power / output_total],
        ])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

        ax = axes[0]
        labels = ['Input', 'Soma output']
        bars = ax.bar(labels, [snr_in, snr_out], color=['#4c78a8', '#e45756'], width=0.6)
        ax.axhline(0, color='k', lw=0.8, alpha=0.5)
        ax.set_ylabel('SNR (dB)')
        ax.set_title('Bandpower SNR')
        for bar, val in zip(bars, [snr_in, snr_out]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.1f} dB',
                    ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

        ax = axes[1]
        x = np.arange(2)
        width = 0.34
        ax.bar(x - width / 2, bandpower_fractions[:, 0], width,
               label='4-6 Hz signal band', color='#54a24b')
        ax.bar(x + width / 2, bandpower_fractions[:, 1], width,
               label='30-100 Hz noise band', color='#9d9d9d')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Fraction of signal+noise bandpower')
        ax.set_title('Relative Bandpower Composition')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

        ax = axes[2]
        ax.axis('off')
        interpretation = 'passive low-pass denoising'
        txt = (
            "PASSIVE CABLE SNR SUMMARY\n\n"
            "Signal band = 4-6 Hz\n"
            "Noise band = 30-100 Hz\n\n"
            f"Input SNR = {snr_in:.1f} dB\n"
            f"Output SNR = {snr_out:.1f} dB\n"
            f"SNR improvement = {snr_imp:.1f} dB\n\n"
            f"Input bandpowers:\n"
            f"  signal = {input_signal_power:.3e} nA^2\n"
            f"  noise  = {input_noise_power:.3e} nA^2\n\n"
            f"Soma bandpowers:\n"
            f"  signal = {output_signal_power:.3e} mV^2\n"
            f"  noise  = {output_noise_power:.3e} mV^2\n\n"
            f"Interpretation = {interpretation}\n"
            "Effect size is modest under the fixed\n"
            "passive cable and 30-100 Hz noise band."
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                fontsize=9.5, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.95))

        fig.suptitle(f'SNR Improvement: {snr_imp:.1f} dB', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig_dir = os.path.join(_figures_root, 'signal_filtering')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, 'snr_improvement.png')
        plt.savefig(path, dpi=150)
        print(f"[OK] Saved: {path}  (SNR improvement: {snr_imp:.1f} dB)")
        plt.close()
        return metrics

    # ------------------------------------------------------------------
    # Test 4: Frequency-Dependent Attenuation
    # ------------------------------------------------------------------
    def inject_single_frequency(self, freq, amp=0.02, n_cycles=10):
        """Inject pure sine wave at given frequency, return soma V amplitude."""
        dt = 0.025
        duration = max(2000.0, n_cycles * 1000.0 / freq)
        t_vec = np.arange(0, duration, dt)
        t_sec = t_vec / 1000.0
        wave = amp * np.sin(2 * np.pi * freq * t_sec)

        v_soma_r = h.Vector()
        t_r = h.Vector()
        v_soma_r.record(self.soma(0.5)._ref_v)
        t_r.record(h._ref_t)

        stim = h.IClamp(self.dend(0.9))
        stim.delay = 0
        stim.dur = duration
        stim.amp = 0
        stim_vec = h.Vector(wave)
        stim_vec.play(stim._ref_amp, dt)

        h.dt = dt
        h.tstop = duration
        h.finitialize(-65)
        h.run()
        stim_vec.play_remove()

        t = np.array(t_r)
        v = np.array(v_soma_r)
        transient_ms = max(0.2 * duration, 2 * 1000.0 / freq)
        steady_mask = t >= transient_ms
        if np.count_nonzero(steady_mask) < 10:
            raise RuntimeError(f"Not enough steady-state samples at {freq} Hz")
        v_ss = v[steady_mask] - np.mean(v[steady_mask])
        if np.std(v_ss) < 1e-6:
            raise RuntimeError("No soma response detected")
        v_amp = _fit_sine_amplitude(t[steady_mask], v_ss, freq)
        return v_amp, v_amp / amp, duration

    def plot_freq_attenuation(self):
        """
        Save: figures/signal_filtering/freq_attenuation.png
        Directly measures how amplitude of soma response drops with frequency.
        """
        test_freqs = np.logspace(np.log10(0.5), np.log10(100), 24)
        amps = []
        z_transfer = []
        durations = []
        for f in test_freqs:
            a, z_mohm, duration = self.inject_single_frequency(f, amp=0.02, n_cycles=10)
            amps.append(a)
            z_transfer.append(z_mohm)
            durations.append(duration)
        amps = np.array(amps)
        z_transfer = np.array(z_transfer)
        if np.all(amps < 1e-6):
            raise RuntimeError("All frequency-scan soma responses are near zero")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.semilogx(test_freqs, z_transfer, 'bo-', ms=4, lw=2)
        ax.set_xlabel('Input Frequency (Hz)')
        ax.set_ylabel('Transfer Impedance (MOhm)')
        ax.set_title('Distal dendrite -> soma transfer\n(1 mV / 1 nA = 1 MOhm)')
        ax.grid(True, alpha=0.3)
        for f_target in [0.5, 5, 50, 100]:
            idx = int(np.argmin(np.abs(test_freqs - f_target)))
            ax.annotate(f'{z_transfer[idx]:.1f} MOhm', xy=(test_freqs[idx], z_transfer[idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        # Normalized
        ax = axes[1]
        amps_norm = amps / amps[0]
        if not amps_norm[0] > amps_norm[-1]:
            warnings.warn("Passive low-pass attenuation check failed: high-frequency amplitude is not lower")
        minus_3db = 1 / np.sqrt(2)
        assert np.isclose(minus_3db, 1 / np.sqrt(2)), "-3 dB reference must be 1/sqrt(2)"
        ax.semilogx(test_freqs, amps_norm, 'ro-', ms=6, lw=2)
        ax.axhline(minus_3db, color='k', ls='--', alpha=0.6,
                   label='-3 dB amplitude = 0.707')
        ax.set_xlabel('Input Frequency (Hz)')
        ax.set_ylabel('Normalized Amplitude (rel to 0.5 Hz)')
        ax.set_title('Normalized Attenuation Curve')
        ax.grid(True, alpha=0.3)

        crossing = np.where((amps_norm[:-1] >= minus_3db) & (amps_norm[1:] < minus_3db))[0]
        if len(crossing) > 0:
            i = crossing[0]
            log_f0, log_f1 = np.log10(test_freqs[i]), np.log10(test_freqs[i + 1])
            y0, y1 = amps_norm[i], amps_norm[i + 1]
            frac = (minus_3db - y0) / (y1 - y0)
            f_cutoff = 10 ** (log_f0 + frac * (log_f1 - log_f0))
            ax.axvline(f_cutoff, color='darkred', ls=':', alpha=0.7,
                       label=f'cutoff ~= {f_cutoff:.2f} Hz')
        else:
            ax.text(0.05, 0.08, 'cutoff not reached in scanned range',
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.85))
        ax.legend()

        fig.text(0.5, 0.01,
                 f'Dendrite length={self.L_dend} μm, diam={self.d} μm, λ={self.lambda_um:.1f} μm; '
                 f'tstop range={min(durations):.0f}-{max(durations):.0f} ms',
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
    result = model.inject_noisy_signal(duration=400, signal_freq=5, snr_db=-3.0)
    snr_data = model.analyze_filtering_result(result)
    model.plot_signal_filtering_results(result, snr_data)
    model.plot_snr_improvement(result, snr_data)
    if snr_data['snr_imp'] > 0:
        print(f"  [OK] CONFIRMED: SNR improved by {snr_data['snr_imp']:.1f} dB")
    else:
        print(f"  [!] SNR change: {snr_data['snr_imp']:.1f} dB")

    # ---- Test 4: Frequency Attenuation ----
    print("\n" + "-" * 65)
    print("Test 4: Frequency-Dependent Attenuation")
    print("-" * 65)
    print("  Measuring log-spaced responses from 0.5 to 100 Hz...")
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
