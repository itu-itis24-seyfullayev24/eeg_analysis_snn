import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import TverskyLoss
import torch.nn.functional as F

class FiringRateRegularizer:
    def __init__(self, model, target_rate=0.05, lambda_reg=0.1):

        self.target_rate = target_rate
        self.lambda_reg = lambda_reg
        self.layer_outputs = {}
        self.hooks = []
        
        self._register_hooks(model)

    def _register_hooks(self, model):
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor) and output.requires_grad:
                     self.layer_outputs[name] = output
            return hook

        for name, layer in model.named_modules():
            if "ALIF" in str(type(layer)): 
                if hasattr(layer, 'return_mem') and layer.return_mem:
                    continue 
                self.hooks.append(layer.register_forward_hook(get_activation(name)))

    def compute_tax(self):
        reg_loss = 0
        for _, spikes in self.layer_outputs.items():
            firing_rate = torch.mean(spikes) 
            reg_loss += (firing_rate - self.target_rate) ** 2

        self.layer_outputs = {}
        
        return self.lambda_reg * reg_loss

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


class TopKClassificationLoss(nn.Module):
    def __init__(self, k_percent=0.05):
        super(TopKClassificationLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.k_percent = k_percent
        self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, inputs, targets_class):
        B, C, H, W = inputs.shape
        flat_inputs = inputs.view(B, C, -1)
        
        k = max(1, int(H * W * self.k_percent))

        top_k_values, _ = torch.topk(flat_inputs, k, dim=2)
        peak_logits = torch.mean(top_k_values, dim=2)
        safe_scale = F.softplus(self.scale)
        peak_logits_scaled = peak_logits * safe_scale

        loss = self.cross_entropy(peak_logits_scaled, targets_class)

        return loss


class FullHybridLoss(nn.Module):
    def __init__(self, smooth=0., lambda_seg=1., lambda_bce=1., lambda_class=0.1, alpha=0.5, beta=0.5, label_smooth=0.):
        super().__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice_loss = TverskyLoss(mode="multilabel", smooth=smooth, from_logits=True, alpha=alpha, beta=beta)
        self.class_loss = TopKClassificationLoss()
        self.lambda_class = lambda_class
        self.lambda_seg = lambda_seg
        self.lambda_bce = lambda_bce
        self.label_smooth = label_smooth
    
    def add_fire_rate_loss(self, model, lambda_fire=0.1, target_rate=0.05):
        self.fire_loss = FiringRateRegularizer(model, target_rate=target_rate, lambda_reg=lambda_fire)

    def forward(self, inputs, targets_mask, targets_class):
        segmentation_loss, classification_loss, bce_loss = 0, 0, 0
        if self.lambda_seg > 0:
            segmentation_loss = self.dice_loss(inputs, targets_mask)
        if self.lambda_class > 0:
            classification_loss = self.class_loss(inputs, targets_class)
        with torch.no_grad():
            smooth_targets = targets_mask * (1 - self.label_smooth) + 0.5 * self.label_smooth
        if self.lambda_bce > 0:
            bce_loss = self.bce_loss(inputs, smooth_targets)

        total_loss = self.lambda_seg * segmentation_loss + self.lambda_class * classification_loss + self.lambda_bce * bce_loss
        
        if hasattr(self, 'fire_loss'):
            tax_loss = self.fire_loss.compute_tax()
            total_loss += tax_loss
        
        return total_loss