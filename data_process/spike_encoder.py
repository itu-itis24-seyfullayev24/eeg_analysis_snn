import torch
import torch.nn as nn
import numpy as np
import mne


class TopographicEncoder(nn.Module):
    def __init__(self, coords, W = 32, H = 32):
        super.__init__()
        self.cords = coords
        self.W = H
        self.H = W

    def forward(self, band_power_62x5):
        band_power_62x5 = band_power_62x5.detach().cpu().numpy()

        topo_maps = []

        for b in range(5):
            power_vec = band_power_62x5[:, b]

            img, _ = mne.viz.plot_topomap(
                power_vec,
                self.coords,
                extrapolate='local',
                image_interp='cubic',
                contours=0,
                outlines='head',
                sphere=(0, 0, 0, 0.1),
                show=False
            )

            img = torch.tensor(img[:, :, 0]).float()

            topo_maps.append(img)

        topo_maps = torch.stack(topo_maps)

        return topo_maps






class SpikeEncoder(nn.Module):
    def __init__(self, t = 30, mode="latency"):
        super().__init__()
        self.t = t
        self.mode = mode.lower()

    def forward(self, topo_5xwxh):
        topo = topo_5xwxh
        topo = (topo - topo.min()) / (topo.max() - topo.min() + 1e-8)  # normalize to [0,1]
        w, h = topo.shape[1], topo.shape[2]


        if self.mode == "rate":
            rand_tensor = torch.rand(self.t, 5, w, h)
            spikes = (rand_tensor < topo).float()
            return spikes


        elif self.mode == "latency":
            spikes = torch.zeros(self.t, 5, w, h)
            latency = (1 - topo) * (self.T - 1)

            for b in range(5):
                for i in range(w):
                    for j in range(h):
                        t = int(latency[b, i, j])
                        spikes[t, b, i, j] = 1
            return spikes

        else:
            raise ValueError(f"unknown spike encoding mode: {self.mode}")
        pass