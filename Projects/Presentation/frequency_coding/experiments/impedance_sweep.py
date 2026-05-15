"""
Impedance Sweep Experiments
Calls the models directly for a quick overview.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.passive_cable import PassiveCable
import numpy as np


def main():
    print("=" * 60)
    print("Impedance Sweep: Quick Verification")
    print("=" * 60)

    model = PassiveCable(L_dend=500, nseg=50, d=2.0)

    print("\nMeasuring input impedance (0.1-100 Hz)...")
    freqs, z_soma, z_dend = model.measure_input_impedance()

    print(f"\n{'Freq (Hz)':>12} {'Z_soma (MΩ)':>14} {'Z_dend (MΩ)':>14}")
    print("-" * 42)
    for f in [0.5, 1, 5, 10, 50, 100]:
        idx = np.argmin(np.abs(freqs - f))
        print(f"{freqs[idx]:>10.1f} Hz {z_soma[idx]:>12.1f} {z_dend[idx]:>12.1f}")

    z_1 = z_soma[np.argmin(np.abs(freqs - 1))]
    z_50 = z_soma[np.argmin(np.abs(freqs - 50))]
    print(f"\nRatio Z(1Hz)/Z(50Hz) = {z_1/z_50:.2f}x")
    if z_1 > z_50:
        print("[OK] Low-pass behavior confirmed")


if __name__ == '__main__':
    main()
