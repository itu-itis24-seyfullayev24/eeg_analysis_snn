import torch
import torch.nn as nn


class WelchModule(nn.Module):
        def __init__(self, window_size, step_size, fs):
            self.window_size = window_size
            self.step_size = step_size
            self.fs = fs
            # you should add the Hann window constructor here
            super(WelchModule, self).__init__()


        def create_windows(eeg, window_size = 200, step_size = 100):
            windows = []

            channels, total_samples = eeg.shape
            assert channels == 32

            for start in range(0, total_samples-window_size+1, step_size):

                end = start + window_size
                window = eeg[:, start:end]
                windows.append(window)

            windows = torch.stack(windows)

            return windows

        # Write code for the welch transformer.  def apply_welch(eeg and other params):





