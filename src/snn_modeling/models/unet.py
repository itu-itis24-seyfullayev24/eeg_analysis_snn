import torch
import torch.nn as nn
import snntorch as snn
from .decoders import ResNetDecoder, SpikingResNetDecoder
from ..layers.stem import BottleneckBlock, ClassifierHead

class SpikingUNet(nn.Module):
    def __init__(self, encoder, in_channels, num_classes, spike_model=snn.Leaky, **neuron_params):
        super(SpikingUNet, self).__init__()

        snn_params = neuron_params.copy()
        snn_params['init_hidden'] = True

        self.encoder = encoder(in_channels, spike_model=spike_model, **snn_params)
        self.bottleneck = BottleneckBlock(512, spike_model=spike_model, **snn_params)
        self.decoder = SpikingResNetDecoder(spike_model=spike_model, **snn_params)
        self.classifier = ClassifierHead(64, num_classes)

    def forward(self, x):
        num_timesteps = x.shape[0]
        logit_rec = []
        for step in range(num_timesteps):
            x_step = x[step, :, :, :, :]
            x_step, skips = self.encoder(x_step)
            x_step = self.bottleneck(x_step)
            x_step = self.decoder(x_step, skips)
            x_step = self.classifier(x_step)
            logit_rec.append(x_step)
        output_logits = torch.stack(logit_rec, dim=0)
        return output_logits

class UNet(nn.Module):
    def __init__(self, encoder, in_channels, num_classes):
        super(UNet, self).__init__()

    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN UNet.")
