
import snntorch as snn
from ..models.unet import SpikingUNet, UNet
from ..models.encoders import ResNet18Encoder, ResNet34Encoder, SpikingResNet18Encoder, SpikingResNet34Encoder

SPIKE_MODEL_MAP = {
    "snn.Leaky": snn.Leaky,
    "snn.Synaptic": snn.Synaptic,
    "snn.Alpha": snn.Alpha
}

def build_model(config):
    model_type = config['model']['type']
    
    if model_type == "SpikingUNet":
        if config['model']['encoder_type'] == "ResNet18":
            encoder = SpikingResNet18Encoder
        elif config['model']['encoder_type'] == "ResNet34":
            encoder = SpikingResNet34Encoder
        else:
            raise ValueError(f"Unknown encoder type: {config['model']['encoder_type']}")
        spike_model_class = SPIKE_MODEL_MAP[config['neuron_params']['spike_model']]

        model = SpikingUNet(
            encoder=encoder,
            in_channels=config['model']['in_channels'],
            num_classes=config['model']['num_classes'],
            spike_model=spike_model_class,
            beta=config['neuron_params']['beta'],
            alpha=config['neuron_params'].get('alpha', None),
            threshold=config['neuron_params']['threshold']
        )
        
    elif model_type == "UNet":
        if config['model']['encoder_type'] == "ResNet18":
            encoder = ResNet18Encoder
        elif config['model']['encoder_type'] == "ResNet34":
            encoder = ResNet34Encoder
        else:
            raise ValueError(f"Unknown encoder type: {config['model']['encoder_type']}")
        spike_model_class = SPIKE_MODEL_MAP[config['neuron_params']['spike_model']]

        model = UNet(
            encoder=encoder,
            in_channels=config['model']['in_channels'],
            num_classes=config['model']['num_classes']
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model