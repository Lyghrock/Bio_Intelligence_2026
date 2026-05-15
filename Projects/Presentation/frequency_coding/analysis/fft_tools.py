"""
FFT and Frequency Analysis Tools
For neural signal frequency analysis
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, fftfreq


def compute_power_spectrum(voltage, fs=40000, nperseg=1024):
    """
    Compute power spectral density using Welch's method

    Parameters:
    -----------
    voltage : array - voltage trace
    fs : float - sampling frequency in Hz
    nperseg : int - FFT segment length

    Returns:
    --------
    freqs : array - frequency bins
    psd : array - power spectral density
    """
    freqs, psd = sig.welch(voltage, fs=fs, nperseg=nperseg)
    return freqs, psd


def bandpass_filter(data, f_low, f_high, fs=40000, order=4):
    """
    Apply bandpass filter to extract frequency band

    Parameters:
    -----------
    data : array - input signal
    f_low, f_high : float - lower and upper frequency bounds (Hz)
    fs : float - sampling frequency
    order : int - filter order

    Returns:
    --------
    filtered : array - filtered signal
    """
    nyq = fs / 2
    low = f_low / nyq
    high = f_high / nyq

    b, a = sig.butter(order, [low, high], btype='band')
    filtered = sig.filtfilt(b, a, data)

    return filtered


def compute_spectrogram(voltage, fs=40000, window='hann',
                        nperseg=256, noverlap=None):
    """
    Compute time-frequency representation

    Parameters:
    -----------
    voltage : array - voltage trace
    fs : float - sampling frequency
    window : str - window function
    nperseg : int - segment length
    noverlap : int - overlap between segments

    Returns:
    --------
    freqs, times, Sxx : frequency, time, and power arrays
    """
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, times, Sxx = sig.spectrogram(
        voltage, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap
    )

    return freqs, times, Sxx


def find_resonance_frequency(freqs, psd, f_range=(1, 50)):
    """
    Find resonance frequency (peak in power spectrum)

    Parameters:
    -----------
    freqs, psd : arrays - frequency and power spectrum
    f_range : tuple - frequency range to search

    Returns:
    --------
    f_peak : float - peak frequency
    power_peak : float - power at peak
    """
    mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    freqs_band = freqs[mask]
    psd_band = psd[mask]

    if len(psd_band) == 0:
        return None, None

    peak_idx = np.argmax(psd_band)
    return freqs_band[peak_idx], psd_band[peak_idx]


def compute_phase_coherence(phase1, phase2):
    """
    Compute phase coherence between two phase signals

    Parameters:
    -----------
    phase1, phase2 : arrays - phase values in radians

    Returns:
    --------
    kappa : float - circular mean resultant length (0-1)
    mean_phase : float - mean phase difference
    """
    # Phase difference
    phase_diff = phase1 - phase2

    # Mean resultant vector
    n = len(phase_diff)
    r = np.sum(np.exp(1j * phase_diff)) / n
    kappa = np.abs(r)
    mean_phase = np.angle(r)

    return kappa, mean_phase


def compute_impedance_from_zap(t, voltage, current, window_size=100):
    """
    Compute impedance from ZAP protocol

    Parameters:
    -----------
    t : array - time vector (ms)
    voltage : array - voltage response (mV)
    current : array - injected current (nA)
    window_size : int - sliding window size for FFT

    Returns:
    --------
    freqs, impedance : arrays - frequency and impedance magnitude
    """
    # Convert to seconds
    t_sec = t / 1000
    dt = t_sec[1] - t_sec[0]

    # Detrend
    v_detrend = signal.detrend(voltage)
    i_detrend = signal.detrend(current)

    # Sliding window FFT
    n_windows = (len(voltage) - window_size) // (window_size // 2)
    freqs_list = []
    z_mag_list = []

    for i in range(n_windows):
        start = i * (window_size // 2)
        end = start + window_size

        v_window = v_detrend[start:end]
        i_window = i_detrend[start:end]

        # FFT
        v_fft = fft(v_window)
        i_fft = fft(i_window)

        # Frequencies
        freqs = fftfreq(window_size, dt)

        # Only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        v_pos = np.abs(v_fft[pos_mask])
        i_pos = np.abs(i_fft[pos_mask])

        # Impedance Z = V / I
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.where(i_pos > 0, v_pos / i_pos, 0)

        freqs_list.extend(freqs_pos)
        z_mag_list.extend(z)

    return np.array(freqs_list), np.array(z_mag_list)


def calculate_snr(signal, noise):
    """
    Calculate signal-to-noise ratio

    Parameters:
    -----------
    signal : array - signal component
    noise : array - noise component

    Returns:
    --------
    snr_db : float - SNR in decibels
    snr_linear : float - SNR in linear scale
    """
    signal_power = np.var(signal)
    noise_power = np.var(noise)

    snr_linear = signal_power / noise_power if noise_power > 0 else float('inf')
    snr_db = 10 * np.log10(snr_linear)

    return snr_db, snr_linear


def extract_oscillation_envelope(voltage, fs=40000, f_carrier=6):
    """
    Extract envelope of oscillation at specific frequency

    Parameters:
    -----------
    voltage : array - voltage trace
    fs : float - sampling frequency
    f_carrier : float - carrier frequency

    Returns:
    --------
    envelope : array - amplitude envelope
    """
    # Bandpass filter
    f_low = f_carrier * 0.5
    f_high = f_carrier * 1.5
    filtered = bandpass_filter(voltage, f_low, f_high, fs)

    # Hilbert transform for envelope
    analytic_signal = sig.hilbert(filtered)
    envelope = np.abs(analytic_signal)

    return envelope


def coherence_analysis(signal1, signal2, fs=40000, f_range=(4, 12)):
    """
    Compute coherence between two signals in a frequency band

    Parameters:
    -----------
    signal1, signal2 : arrays - input signals
    fs : float - sampling frequency
    f_range : tuple - frequency band to analyze

    Returns:
    --------
    coherence : float - mean coherence in band
    freqs : array - frequency bins
    coh_spectrum : array - coherence spectrum
    """
    # Welch's cross-spectral density
    freqs, csd = sig.csd(signal1, signal2, fs=fs, nperseg=1024)
    _, psd1 = sig.psd(signal1, fs=fs, nperseg=1024)
    _, psd2 = sig.psd(signal2, fs=fs, nperseg=1024)

    # Coherence Cxy = |Pxy|^2 / (Pxx * Pyy)
    coherence = np.abs(csd) ** 2 / (psd1 * psd2)

    # Filter by frequency range
    mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    mean_coherence = np.mean(coherence[mask])

    return mean_coherence, freqs, coherence


# Alias for convenience
signal = sig
