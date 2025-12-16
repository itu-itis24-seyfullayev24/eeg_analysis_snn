import torch
import torch.nn as nn
import snntorch as snn
from .decoders import ResNetDecoder, SpikingResNetDecoder
from ..layers.stem import BottleneckBlock, ClassifierHead
from ..layers.neurons import TimeDistributed
import snntorch.spikegen as spikegen
import torchvision

class SpikingUNet(nn.Module):
    def __init__(self, encoder, in_channels, num_classes, config, spike_model=snn.Leaky, **neuron_params):
        super(SpikingUNet, self).__init__()

        snn_params = neuron_params.copy()
        if spike_model.__name__  != "ALIF":
            snn_params['init_hidden'] = True
        self.encoding = config['data'].get('encoding_method', 'direct')
        self.num_timesteps = config['data'].get('num_timesteps', 10)
        self.encoder = encoder(in_channels, p_drop=config['model'].get('dropout', 0.2), spike_model=nn.SiLU)#spike_model, **snn_params)
        self.bottleneck = BottleneckBlock(512, p_drop=config['model'].get('dropout', 0.2), spike_model=spike_model, **snn_params)
        self.decoder = SpikingResNetDecoder(spike_model=spike_model, **snn_params)
        self.classifier = ClassifierHead(64, num_classes)

    def forward(self, x):
        if self.encoding == 'latency':
            x_static = x.mean(dim=0)
            x = spikegen.latency(x_static, num_steps=self.num_timesteps, tau=5, threshold=0.01, normalize=True, clip=True)

        elif self.encoding == 'rate': # Converges to Poisson encoding
            
            rand_map = torch.rand_like(x) 
            x = (x > rand_map).float()
        elif self.encoding == 'direct':
            pass
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding}")
    
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        logits = self.classifier(x)

        return logits.mean(dim=0) 
    
class UNet(nn.Module):
    def __init__(self, encoder, in_channels, num_classes):
        super(UNet, self).__init__()

    def forward(self, x):
        raise NotImplementedError("This is a placeholder for the ANN UNet.")

class SpikingResNetClassifier(nn.Module):
    def __init__(self, encoder_backbone, num_classes=5):
        super().__init__()

        self.encoder = encoder_backbone 
        self.num_classes = num_classes
        self.classifier = ClassifierHead(512, num_classes)

    def forward(self, x):
        
        x_static = x.mean(dim=0)
        #x = spikegen.latency(x_static, num_steps=16, tau=2, threshold=0.01, normalize=False, clip=True,  bypass=True)
        features, _ = self.encoder(x)
        out = self.classifier(features)

        return out.mean(dim=0)  # Mean over time dimension

class PlainResNetClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Standard ResNet18, no spikes
        self.backbone = torchvision.models.resnet18(pretrained=False)
        
        # Adjust first layer for 62 channels (if using 62 ch input)
        # OR 5 channels (if using Bands as channels)
        # Assuming input is (Batch, 5, 32, 32)
        self.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the FC layer
        self.backbone.fc = nn.Identity()
        
        # The Smasher Head (Linear)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (Batch, Time, 5, 32, 32)
        # For ANN, we can just take the MEAN over time to make it a static image
        # OR just take the first frame if they are identical.
        
        if x.ndim == 5:
            x = torch.mean(x, dim=1) # Collapse time -> (B, 5, 32, 32)
            
        features = self.backbone(x) # -> (B, 512)
        logits = self.head(features) # -> (B, 5)
        return logits