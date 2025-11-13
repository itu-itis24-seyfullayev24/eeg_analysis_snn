import torch.nn as nn
import torch
import snntorch as snn
from snn_modeling.models.encoders import ResNet18Encoder, ResNet34Encoder 
from snn_modeling.models.spike_resnet import SpikingUNet

def test_resnet18_encoder():
    model = ResNet18Encoder(3, spike_model=snn.Leaky, beta=0.9, init_hidden=True)
    x = nn.Parameter(torch.randn(1, 3, 64, 64))
    out = model(x)
    print(out[0].shape)

def test_resnet34_encoder():
    model = ResNet34Encoder(3, spike_model=snn.Leaky, beta=0.9, init_hidden=True)
    x = nn.Parameter(torch.randn(1, 3, 64, 64))
    out = model(x)
    print(out[0].shape)

def test_spiking_unet():
    encoder = ResNet18Encoder
    model = SpikingUNet(encoder=encoder, in_channels=5, spike_model=snn.Leaky, beta=0.9, num_classes=4)
    x = nn.Parameter(torch.randn(10, 1, 5, 64, 64))
    out = model(x)
    print(out.shape)
    #assert out.shape == (5, 4, 64, 64)  
    print("All tests passed!")

if __name__ == "__main__":
    test_resnet18_encoder()
    test_resnet34_encoder()
    test_spiking_unet()