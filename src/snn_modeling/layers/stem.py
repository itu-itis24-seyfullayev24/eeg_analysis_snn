import torch.nn as nn
import torch
import snntorch as snn
from .residual_blocks import ConvSpiking
from .neurons import TimeDistributed
import torch.nn.functional as F

class StemLayer(nn.Module):
    def __init__(self, in_channels):
        super(StemLayer, self).__init__()
        
        self.layer = ConvSpiking(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            spike_model=nn.Identity,
            use_norm=False)
        
        self.norm = TimeDistributed(nn.InstanceNorm2d(64, affine=True,eps=1e-6))
        self.act = nn.SiLU()
    def forward(self, x):

        return self.act(self.norm(self.layer(x)))
    
class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes, kernel_size=1):
        super(ClassifierHead, self).__init__()

        self.head = TimeDistributed(nn.Conv2d(in_features, num_classes, kernel_size=kernel_size, bias=True))
    
    def forward(self, x):
        return self.head(x)
        
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, p_drop=0.2, spike_model=snn.Leaky, **neuron_params):
        super(BottleneckBlock, self).__init__()
        self.conv1 = ConvSpiking(in_channels, in_channels // 2, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)
        self.drop = TimeDistributed(nn.Dropout2d(p=p_drop))
        self.conv2 = ConvSpiking(in_channels // 2, in_channels, kernel_size=3, padding=1, spike_model=spike_model, **neuron_params)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, feature_dim=512, head_dim=128):
        super(ProjectionHead, self).__init__()
        self.supcon_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_dim, head_dim, kernel_size=1)
        )
    def forward(self, features):
        T, B, C, H, W = features.shape
        proj = self.supcon_head(features.view(T * B, C, H, W)) + 1e-6 
        embedding = F.normalize(proj.view(T * B, -1), dim=1)
        return embedding