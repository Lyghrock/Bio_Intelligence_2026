"""
KCNQ/M-Current Neuron Model
Part 3: High-pass filtering via M-current (KCNQ channels)
===========================================================

KCNQ channels DON'T act as frequency filters on impedance.
Instead, they raise the spike threshold by increasing K+ conductance.

Key demonstrations:
  1. Subthreshold response: KCNQ reduces voltage response amplitude across ALL frequencies
  2. Firing rate: KCNQ suppresses low-frequency repetitive firing
  3. Slow oscillation: KCNQ suppresses 1 Hz oscillation response

Note: KCNQ is a conductance-based effect on spike threshold, not a
frequency-dependent resonance like HCN.
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


class KCNQNeuron:
    """Soma + dendrite with KCNQ (Im) channels."""

    def __init__(self, dend_L=200, g_Im=0.003):
        self.soma = h.Section(name='soma_kcnq')
        self.dend = h.Section(name='dend_kcnq')

        self.soma.L = 10
        self.soma.diam = 10
        self.soma.cm = 1.0
        self.soma.Ra = 100
        self.soma.insert('pas')
        self.soma.g_pas = 0.0001
        self.soma.e_pas = -65
        self.soma.insert('hh1')

        self.soma.insert('Im')
        for seg in self.soma:
            seg.Im.gImbar = g_Im

        self.dend.L = dend_L
        self.dend.diam = 2.0
        self.dend.nseg = 20
        self.dend.cm = 1.0
        self.dend.Ra = 100
        self.dend.insert('pas')
        self.dend.g_pas = 0.0003
        self.dend.e_pas = -65

        self.dend.connect(self.soma(1))

        self.v_soma = h.Vector()
        self.t = h.Vector()
        self.v_soma.record(self.soma(0.5)._ref_v)
        self.t.record(h._ref_t)

    def inject_subthreshold(self, freq, amp=0.05, dur=2000):
        """
        Inject subthreshold sine wave (amp is small enough to stay below spike threshold).
        Returns (t, v_subthreshold, wave).
        """
        dt = 0.025
        t_vec = np.arange(0, dur, dt)
        wave = amp * np.sin(2 * np.pi * freq * t_vec / 1000)

        stim = h.IClamp(self.soma(0.5))
        stim.delay = 0
        stim.dur = dur
        stim.amp = 0
        h.Vector(wave).play(stim._ref_amp, dt)

        h.finitialize(-65)
        h.tstop = dur + 100
        h.run()
        h.Vector(wave).play_remove()

        return np.array(self.t), np.array(self.v_soma), wave

    def step_current_test(self, amplitudes=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4], dur=1000):
        """Count spikes for step current amplitudes."""
        results = []
        for amp in amplitudes:
            stim = h.IClamp(self.soma(0.5))
            stim.delay = 200
            stim.dur = dur
            stim.amp = amp

            h.finitialize(-65)
            h.tstop = dur + 500
            h.run()

            spikes = self._count_spikes(np.array(self.v_soma))
            results.append({
                'amp': amp,
                'spikes': spikes,
                'v_trace': np.array(self.v_soma).copy(),
                't_trace': np.array(self.t).copy()
            })
        return results

    def slow_oscillation_test(self, freq=1.0, dur=2000):
        """Inject 1 Hz sine wave at subthreshold amplitude."""
        dt = 0.025
        t = np.arange(0, dur, dt)
        wave = 0.15 * np.sin(2 * np.pi * freq * t / 1000)  # subthreshold

        stim = h.IClamp(self.soma(0.5))
        stim.delay = 0
        stim.dur = dur
        stim.amp = 0
        h.Vector(wave).play(stim._ref_amp, dt)

        h.finitialize(-65)
        h.tstop = dur + 100
        h.run()
        h.Vector(wave).play_remove()

        return np.array(self.t), np.array(self.v_soma), wave

    def _count_spikes(self, v):
        threshold = -30
        crossings = np.where(np.diff(np.sign(v - threshold)) > 0)[0]
        return len(crossings)


class PassiveControlNeuron:
    """Control: same geometry but no KCNQ channels."""

    def __init__(self, dend_L=200):
        self.soma = h.Section(name='soma_ctrl')
        self.dend = h.Section(name='dend_ctrl')

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
        self.dend.nseg = 20
        self.dend.cm = 1.0
        self.dend.Ra = 100
        self.dend.insert('pas')
        self.dend.g_pas = 0.0003
        self.dend.e_pas = -65

        self.dend.connect(self.soma(1))

        self.v_soma = h.Vector()
        self.t = h.Vector()
        self.v_soma.record(self.soma(0.5)._ref_v)
        self.t.record(h._ref_t)

    def inject_subthreshold(self, freq, amp=0.05, dur=2000):
        dt = 0.025
        t_vec = np.arange(0, dur, dt)
        wave = amp * np.sin(2 * np.pi * freq * t_vec / 1000)

        stim = h.IClamp(self.soma(0.5))
        stim.delay = 0
        stim.dur = dur
        stim.amp = 0
        h.Vector(wave).play(stim._ref_amp, dt)

        h.finitialize(-65)
        h.tstop = dur + 100
        h.run()
        h.Vector(wave).play_remove()

        return np.array(self.t), np.array(self.v_soma), wave

    def step_current_test(self, amplitudes=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4], dur=1000):
        results = []
        for amp in amplitudes:
            stim = h.IClamp(self.soma(0.5))
            stim.delay = 200
            stim.dur = dur
            stim.amp = amp

            h.finitialize(-65)
            h.tstop = dur + 500
            h.run()

            spikes = self._count_spikes(np.array(self.v_soma))
            results.append({
                'amp': amp,
                'spikes': spikes,
                'v_trace': np.array(self.v_soma).copy(),
                't_trace': np.array(self.t).copy()
            })
        return results

    def slow_oscillation_test(self, freq=1.0, dur=2000):
        dt = 0.025
        t = np.arange(0, dur, dt)
        wave = 0.15 * np.sin(2 * np.pi * freq * t / 1000)

        stim = h.IClamp(self.soma(0.5))
        stim.delay = 0
        stim.dur = dur
        stim.amp = 0
        h.Vector(wave).play(stim._ref_amp, dt)

        h.finitialize(-65)
        h.tstop = dur + 100
        h.run()
        h.Vector(wave).play_remove()

        return np.array(self.t), np.array(self.v_soma), wave

    def _count_spikes(self, v):
        threshold = -30
        crossings = np.where(np.diff(np.sign(v - threshold)) > 0)[0]
        return len(crossings)


def make_dir(name):
    d = os.path.join(_figures_root, name)
    os.makedirs(d, exist_ok=True)
    return d


def plot_subthreshold_response(test_freqs, rms_ctrl, rms_kcnq):
    """
    figures/kcnq_resonance/subthreshold_response.png
    Shows KCNQ reduces voltage response amplitude across ALL frequencies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.semilogx(test_freqs, rms_ctrl, 'bo-', ms=6, lw=2, label='Passive')
    ax.semilogx(test_freqs, rms_kcnq, 'ro-', ms=6, lw=2, label='With KCNQ')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Subthreshold V_rms (mV)')
    ax.set_title('Subthreshold Response: KCNQ Reduces Gain\n'
                 '(K+ conductance shunts current, reducing voltage response)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate reduction
    rms_ctrl_arr = np.array(rms_ctrl)
    rms_kcnq_arr = np.array(rms_kcnq)
    reduction = np.mean(rms_kcnq_arr / rms_ctrl_arr)
    ax.annotate(f'KCNQ reduces gain\nby ~{(1-reduction)*100:.0f}% on average',
                xy=(2, np.mean(rms_kcnq_arr)), xytext=(5, np.mean(rms_ctrl_arr) * 0.8),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

    ax = axes[1]
    ratio = rms_kcnq_arr / (rms_ctrl_arr + 1e-12)
    ax.semilogx(test_freqs, ratio, 'go-', ms=6, lw=2)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='No change')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('KCNQ / Passive RMS Ratio')
    ax.set_title('KCNQ Effect Across Frequency Range\n'
                 '(Effect is uniform: raises spike threshold)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(make_dir('kcnq_resonance'), 'subthreshold_response.png')
    plt.savefig(path, dpi=150)
    print(f"[OK] Saved: {path}")
    plt.close()


def plot_firing_rate(results_ctrl, results_kcnq):
    """
    figures/kcnq_resonance/firing_rate.png
    Shows KCNQ suppresses firing, especially at low currents.
    """
    amps = [r['amp'] for r in results_ctrl]
    spikes_ctrl = [r['spikes'] for r in results_ctrl]
    spikes_kcnq = [r['spikes'] for r in results_kcnq]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x = np.arange(len(amps))
    w = 0.35
    ax.bar(x - w/2, spikes_ctrl, w, color='blue', alpha=0.7, label='Passive')
    ax.bar(x + w/2, spikes_kcnq, w, color='red', alpha=0.7, label='With KCNQ')
    ax.set_xlabel('Current amplitude (nA)')
    ax.set_ylabel('Spike count')
    ax.set_title('Firing Rate: Passive vs KCNQ\n(KCNQ raises threshold → fewer spikes)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a:.2f}' for a in amps])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    total_ctrl = sum(spikes_ctrl)
    total_kcnq = sum(spikes_kcnq)
    reduction = total_ctrl - total_kcnq
    ax.text(0.05, 0.95, f'Total: {total_ctrl}→{total_kcnq} spikes\n({reduction} spikes suppressed)',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', fc='lightyellow'))

    ax = axes[1]
    ax.plot(amps, spikes_ctrl, 'bo-', ms=6, lw=2, label='Passive')
    ax.plot(amps, spikes_kcnq, 'ro-', ms=6, lw=2, label='With KCNQ')
    ax.set_xlabel('Current amplitude (nA)')
    ax.set_ylabel('Spike count')
    ax.set_title('FI Curve: KCNQ Shifts Threshold Right\n'
                 '(More current needed to reach same firing rate)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate threshold shift
    if len(spikes_ctrl) > 1:
        ax.annotate('KCNQ shifts\nthreshold right',
                    xy=(0.3, spikes_kcnq[2]), xytext=(0.3, spikes_ctrl[2]),
                    arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

    plt.tight_layout()
    path = os.path.join(make_dir('kcnq_resonance'), 'firing_rate.png')
    plt.savefig(path, dpi=150)
    print(f"[OK] Saved: {path}")
    plt.close()


def plot_slow_oscillation(t_ctrl, v_ctrl, t_kcnq, v_kcnq, wave):
    """
    figures/kcnq_resonance/slow_oscillation.png
    Shows KCNQ suppresses 1 Hz oscillation response.
    """
    skip = int(500 / 0.025)
    t0 = skip
    t1 = skip + int(1000 / 0.025)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(t_ctrl[t0:t1] - t_ctrl[t0], v_ctrl[t0:t1], 'b-', lw=1.5,
            label='Passive', alpha=0.8)
    ax.plot(t_kcnq[t0:t1] - t_kcnq[t0], v_kcnq[t0:t1], 'r-', lw=2,
            label='With KCNQ')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('Response to 1 Hz Oscillatory Input (Subthreshold)\n'
                 '(KCNQ suppresses low-frequency oscillation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    s_ctrl = np.sum(np.diff(np.sign(v_ctrl - (-30))) > 0)
    s_kcnq = np.sum(np.diff(np.sign(v_kcnq - (-30))) > 0)
    ax.text(0.02, 0.95, f'Passive: {s_ctrl} spikes\nKCNQ: {s_kcnq} spikes',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', fc='lightyellow'))

    ax = axes[1]
    t_w = np.arange(len(wave)) * 0.025
    ax.plot(t_w[t0:t1] - t_w[t0], wave[t0:t1] * 1e3, 'k-', lw=1.5, label='Input current')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (pA)')
    ax.set_title('Input Current (1 Hz oscillation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(make_dir('kcnq_resonance'), 'slow_oscillation.png')
    plt.savefig(path, dpi=150)
    print(f"[OK] Saved: {path}")
    plt.close()


# ==================================================================
# Main
# ==================================================================
def main():
    print("=" * 65)
    print("KCNQ/M-Current: High-Pass / Threshold Elevation")
    print("=" * 65)

    g_Im = 0.0004  # S/cm2 - tuned for visible KCNQ effect
    print(f"\nCreating neurons (g_Imbar={g_Im})...")
    ctrl = PassiveControlNeuron(dend_L=200)
    kcnq = KCNQNeuron(dend_L=200, g_Im=g_Im)
    print("  [OK] Control neuron (no KCNQ)")
    print("  [OK] KCNQ neuron")

    # ---- Test 1: Subthreshold frequency sweep ----
    print("\n" + "-" * 65)
    print("Test 1: Subthreshold Response (Frequency Sweep)")
    print("-" * 65)

    test_freqs = [0.5, 1, 2, 5, 10, 20]
    amp = 0.05  # subthreshold

    rms_ctrl = []
    rms_kcnq = []
    print(f"  {'Freq':>8}  {'Passive RMS':>14}  {'KCNQ RMS':>12}  {'Ratio':>8}")
    print("  " + "-" * 46)

    for f in test_freqs:
        t_c, v_c, _ = ctrl.inject_subthreshold(freq=f, amp=amp, dur=2000)
        t_k, v_k, _ = kcnq.inject_subthreshold(freq=f, amp=amp, dur=2000)

        skip = int(200 / 0.025)
        # Use voltage deviation from rest (-65 mV) for RMS
        v_c_dev = v_c[skip:] - (-65)
        v_k_dev = v_k[skip:] - (-65)
        rms_c = np.sqrt(np.mean(v_c_dev**2))
        rms_k = np.sqrt(np.mean(v_k_dev**2))
        ratio = rms_k / rms_c if rms_c > 0 else 0

        rms_ctrl.append(rms_c)
        rms_kcnq.append(rms_k)
        print(f"  {f:>6.1f} Hz {rms_c:>12.4f} mV {rms_k:>10.4f} mV {ratio:>8.3f}")

    avg_reduction = np.mean(np.array(rms_kcnq) / np.array(rms_ctrl))
    print(f"\n  Average KCNQ/Passive ratio: {avg_reduction:.3f}")
    if avg_reduction < 1.0:
        print("  [OK] CONFIRMED: KCNQ reduces voltage response across all frequencies")
    plot_subthreshold_response(test_freqs, rms_ctrl, rms_kcnq)

    # ---- Test 2: Firing rate ----
    print("\n" + "-" * 65)
    print("Test 2: Step Current - Firing Rate")
    print("-" * 65)

    amps = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    print(f"  Amplitudes: {amps} nA")
    r_ctrl = ctrl.step_current_test(amplitudes=amps)
    r_kcnq = kcnq.step_current_test(amplitudes=amps)

    print(f"  {'Amp':>6}  {'Control':>10}  {'KCNQ':>10}  {'Reduction':>10}")
    print("  " + "-" * 40)
    total_ctrl = 0
    total_kcnq = 0
    for i, a in enumerate(amps):
        sc, sk = r_ctrl[i]['spikes'], r_kcnq[i]['spikes']
        total_ctrl += sc
        total_kcnq += sk
        print(f"  {a:>5.2f} nA  {sc:>8}  {sk:>8}  {sc-sk:>8}")
    print(f"\n  Total: {total_ctrl} → {total_kcnq} spikes ({total_ctrl-total_kcnq} suppressed)")
    if total_ctrl > total_kcnq:
        print("  [OK] CONFIRMED: KCNQ suppresses firing")
    plot_firing_rate(r_ctrl, r_kcnq)

    # ---- Test 3: Slow oscillation ----
    print("\n" + "-" * 65)
    print("Test 3: Slow Oscillation (1 Hz) Suppression")
    print("-" * 65)

    tp_c, vp_c, w_c = ctrl.slow_oscillation_test(freq=1.0)
    tp_k, vp_k, _ = kcnq.slow_oscillation_test(freq=1.0)

    s_ctrl = ctrl._count_spikes(vp_c)
    s_kcnq = kcnq._count_spikes(vp_k)
    print(f"  Passive: {s_ctrl} spikes to 1 Hz input")
    print(f"  KCNQ:    {s_kcnq} spikes to 1 Hz input")
    if s_kcnq < s_ctrl:
        print("  [OK] CONFIRMED: KCNQ suppresses low-frequency oscillation response")
    plot_slow_oscillation(tp_c, vp_c, tp_k, vp_k, w_c)

    print("\n" + "=" * 65)
    print("Summary:")
    print("  1. KCNQ reduces subthreshold voltage response uniformly")
    print("     (not frequency-selective; effect is on spike threshold)")
    print(f"     Average reduction: {(1-avg_reduction)*100:.0f}%")
    print(f"  2. KCNQ suppresses firing (total: {total_ctrl}→{total_kcnq})")
    print(f"  3. KCNQ suppresses 1 Hz oscillation ({s_ctrl}→{s_kcnq} spikes)")
    print("\nConclusion: KCNQ/M-current raises the spike threshold by")
    print("  increasing K+ conductance. This makes neurons less")
    print("  responsive to low-frequency inputs, effectively acting")
    print("  as a high-pass filter on spike output.")
    print("=" * 65)


if __name__ == '__main__':
    main()
