import torch
import torch.optim as optim
from src.snn_modeling.utils.loss import DiceBCELoss, FullHybridLoss
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from src.snn_modeling.dataloader.dummy_loader import get_dummy_batch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import snntorch as snn
import segmentation_models_pytorch as smp
import wandb
import warnings

# Ignore the specific sklearn warning about missing classes
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.")
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
def manual_reset(model):
    """
    Recursively reset the hidden state of all SNN layers.
    This ensures the computation graph is broken between epochs.
    """
    for module in model.modules():
        # Check if the module is a spiking neuron
        if isinstance(module, (snn.Leaky, snn.Synaptic, snn.Alpha)):
            if hasattr(module, 'reset_mem'):
                module.reset_mem()
            if hasattr(module, 'reset_hidden'):
                module.reset_hidden()
            if hasattr(module, 'detach_hidden'):
                module.detach_hidden()

def run_training(config, model, device):
    wandb.init(
        project=config['logging']['project_name'],
        name=config['logging']['run_name'],
        config=config,
        tags=config['logging']['tags'],
        mode="disabled" if config['logging'].get('offline') else "online"
    )
    run_name = f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("results", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Initializing TensorBoard: {log_dir}")
    model = model.to(device)
    model.train()
    #wandb.watch(model, log="all", log_freq=100)
    lr = config['training'].get('learning_rate', 1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    loss_fn = FullHybridLoss(
        smooth = config['loss'].get('smooth', 0.0),
        lambda_bce = config['loss'].get('lambda_bce', 1.0),
        lambda_class = config['loss'].get('lambda_class', 1.0)
    )
    inputs, targets, targets_c = get_dummy_batch(config, device)

    epochs = config['training']['epochs']
    print("--- Starting Sanity Check: Overfitting One Batch ---")
    for epoch in range(epochs):
        optimizer.zero_grad()
        manual_reset(model)
        outputs = model(inputs)
        outputs_aggregated = torch.mean(outputs, dim=0)
        loss = loss_fn(outputs_aggregated, targets, targets_c)
        loss.backward()
        optimizer.step()
        val_probs = torch.sigmoid(outputs_aggregated)
        val_targets = (targets > 0.5).long()
        tp, fp, fn, tn = smp.metrics.get_stats(
            val_probs, 
            val_targets, 
            mode='multilabel', 
            threshold=0.5
        )
        dice_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # IoU Score
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # Precision
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # Recall
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        
        val_probs = (val_probs > 0.5).float()
        energy = torch.sum(val_probs, dim=(2, 3))
        pred_c = torch.argmax(energy, dim=1).cpu().numpy()
        true_c = targets_c.cpu().numpy()
    
        acc = accuracy_score(true_c, pred_c)
        bal_acc = balanced_accuracy_score(true_c, pred_c)
        log_dict = {
            "epoch": epoch,
            "Loss/Train": loss.item(),
            "Metrics/Dice": dice_score.item(),
            "Metrics/IoU": iou_score.item(),
            "Metrics/Precision": precision.item(),
            "Metrics/Recall": recall.item(),
            "Metrics/Accuracy": acc,
            "Metrics/Balanced_Accuracy": bal_acc
        }

        writer.add_scalar("Loss/Train", loss.item(), epoch)
        writer.add_scalar("Metrics/Dice", dice_score.item(), epoch)
        writer.add_scalar("Metrics/IoU", iou_score.item(), epoch)
        writer.add_scalar("Metrics/Precision", precision.item(), epoch)
        writer.add_scalar("Metrics/Recall", recall.item(), epoch)
        writer.add_scalar("Metrics/Accuracy", acc, epoch)
        writer.add_scalar("Metrics/Balanced_Accuracy", bal_acc, epoch)
        if (epoch + 1) % config['training'].get('log_interval', 10) == 0:
            num_samples_to_log = min(4, inputs.shape[1]) 
        
            for sample_idx in range(num_samples_to_log):

                true_class_idx = targets_c[sample_idx].item()
                img_input = inputs[:, sample_idx, :, :, :].mean(dim=(0, 1)).cpu().detach().unsqueeze(0).numpy()

                img_target = targets[sample_idx, true_class_idx, :, :].cpu().detach().unsqueeze(0).numpy()
    
                img_pred = val_probs[sample_idx, true_class_idx, :, :].cpu().detach().unsqueeze(0).numpy()

                caption_str = f"Sample {sample_idx} - Class {true_class_idx}"
            
                log_dict[f"Visuals/Sample_{sample_idx}_Input"] = wandb.Image(img_input, caption=caption_str)
                log_dict[f"Visuals/Sample_{sample_idx}_Target"] = wandb.Image(img_target, caption=caption_str)
                log_dict[f"Visuals/Sample_{sample_idx}_Pred"] = wandb.Image(img_pred, caption=caption_str)
                writer.add_image(f"Visuals/Sample_{sample_idx}_Input", img_input, epoch)
                writer.add_image(f"Visuals/Sample_{sample_idx}_Target", img_target, epoch)
                writer.add_image(f"Visuals/Sample_{sample_idx}_Pred", img_pred, epoch)
            
        wandb.log(log_dict, step=epoch)
    
    print("--- Sanity Check Complete ---")
    print(f"Final Loss: {loss.item():.6f}")
    wandb.finish()
    writer.close()