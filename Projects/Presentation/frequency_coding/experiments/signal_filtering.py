"""
Signal Filtering Experiment
Demonstrates passive membrane's natural denoising ability.
Calls the PassiveCable model from models/passive_cable.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.passive_cable import PassiveCable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_figures_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))


def main():
    print("=" * 60)
    print("Signal Filtering Experiment: Passive Denoising")
    print("=" * 60)

    model = PassiveCable(L_dend=500, nseg=50, d=2.0)
    print(f"\nModel: L={model.L_dend} μm, λ={model.lambda_um:.1f} μm")

    # Test different signal frequencies
    print("\nTest: Frequency-dependent transmission")
    print(f"{'Freq (Hz)':>12} {'Soma Amp (mV)':>18}")
    print("-" * 32)

    for f in [1, 2, 5, 10, 20, 40]:
        t, v_soma, v_dend = model.inject_sinusoidal(freq=f, amp=0.1, duration=1500)
        skip = int(200 / 0.025)
        v_ss = v_soma[skip:]
        amp = (np.max(v_ss) - np.min(v_ss)) / 2
        print(f"{f:>10.0f} Hz {amp:>16.2f}")

    # SNR improvement test
    print("\nTest: SNR improvement (5 Hz signal + 30-100 Hz noise)")
    result = model.inject_noisy_signal(duration=3000, signal_freq=5, snr_db=-3.0)
    snr_data = model.plot_snr_improvement(result)

    print(f"\n  Input SNR:  {snr_data['snr_in']:.1f} dB")
    print(f"  Output SNR: {snr_data['snr_out']:.1f} dB")
    print(f"  Improvement: {snr_data['snr_imp']:.1f} dB")

    if snr_data['snr_imp'] > 0:
        print("\n  [OK] CONFIRMED: Passive cable improves SNR")
    print("\nConclusion: Passive membrane capacitance naturally filters")
    print("  high-frequency noise, preserving low-frequency signals.")


if __name__ == '__main__':
    main()
