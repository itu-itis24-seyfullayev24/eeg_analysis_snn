import torch
import torch.nn as nn
import pywt

class WaveletModule(nn.Module):
    def __init__(self, window_size=256, step_size=128, fs=200):
        super(WaveletModule, self).__init__()
        self.window_size = window_size
        self.step_size = step_size
       	self.fs = fs
        self.wavelet_name = "morl"

    def create_windows(self, eeg, window_size=256, step_size=128):
        windows = []
        channels, total_samples = eeg.shape
        assert channels == 14

        for start in range(0, total_samples - window_size + 1, step_size):
            end = start + window_size
            window = eeg[:, start:end]
            windows.append(window)

        return torch.stack(windows)

    def wavelet(self, window, channels=14):
        scales = torch.arange(1, 64)
        coeffs = []

        for channel in range(channels):
            cwtmatr, freqs = pywt.cwt(
                window[channel].cpu().numpy(),
                scales.cpu().numpy(),
                self.wavelet_name,
                sampling_period=1 / self.fs
            )
            coeffs.append(torch.tensor(cwtmatr).float())

        return torch.stack(coeffs)

    def bandpowers_wavelet(self, coeffs):
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 14),
            "beta": (14, 31),
            "gamma": (31, 50)
        }

        num_scales = coeffs.shape[1]
        wavelet_freqs = torch.linspace(1, 100, num_scales)

        band_powers = []

        for band_name, (low, high) in bands.items():
            idx = (wavelet_freqs >= low) & (wavelet_freqs <= high)
            power = coeffs[:, idx, :].pow(2).mean(dim=(1, 2))
            band_powers.append(power)

        return torch.stack(band_powers, dim=1)
