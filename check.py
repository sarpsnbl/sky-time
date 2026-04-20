import torch
from config import Config as cfg
from main import TimeOfDayModel, CyclicMSELoss, cyclic_mae_minutes, load_checkpoint
from TimeOfDayDataLoader import TimeOfDayDataset, create_dataloaders, get_transforms

device = torch.device("cuda")

model = TimeOfDayModel(pretrained=False, freeze_until=cfg.FREEZE_UNTIL,
                       hidden_dim=cfg.HIDDEN_DIM, dropout=cfg.DROPOUT).to(device)
model.to(memory_format=torch.channels_last)
load_checkpoint("checkpoints/best_fold2.pt", model, device=device)
model.eval()

dataset = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=get_transforms(augment=False))
_, val_loader = create_dataloaders(dataset, dataset, fold=2, n_splits=cfg.N_SPLITS,
                                   batch_size=cfg.BATCH_SIZE, num_workers=0,
                                   val_ratio=cfg.VAL_RATIO)

criterion = CyclicMSELoss()
total_loss = total_mae = 0.0
with torch.no_grad():
    for images, metadata, targets in val_loader:
        images = images.to(device, memory_format=torch.channels_last)
        metadata, targets = metadata.to(device), targets.to(device)
        preds = model(images, metadata)
        total_loss += criterion(preds, targets).item()
        total_mae += cyclic_mae_minutes(preds.cpu(), targets.cpu())

n = len(val_loader)
print(f"Fold 0 single model — loss: {total_loss/n:.6f}  MAE: {float(total_mae/n):.2f} min")