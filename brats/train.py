import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import MRIDataset
from decoder import UNet3DDecoder
from augmentations import combine_aug
from tqdm import tqdm
from functools import partial
from torch.optim import AdamW

import sys
import os
import pdb
from torchsummary import summary
import segmentation_models_pytorch as smp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# pdb.set_trace()
from timesformer.models.vit import VisionTransformer

batch_size = 1
train_dataset = MRIDataset("../BraTS2020", combine_aug, "train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_batch_size = 1
valid_dataset = MRIDataset("../BraTS2020", combine_aug, "val")
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

test_batch_size = 1
test_dataset = MRIDataset("../BraTS2020", combine_aug, "test")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Segmentformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, num_classes=128, num_frames=128, attention_type='divided_space_time',  pretrained_model=''):
        super(Segmentformer, self).__init__()
        self.pretrained=True
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, in_chans=4, patch_size=patch_size, embed_dim=768, depth=4, num_heads=4, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type,)
        self.model.head = UNet3DDecoder()

        self.attention_type = attention_type
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
    
    def forward(self, x):
        x = self.model(x)
        return x

# Training loop
def train_model(model, optimizer, total_loss, train_loader, val_loader, num_epochs=1, device=torch.device('cpu')):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # pdb.set_trace()
            masks = torch.argmax(masks, dim=1)
            loss = total_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            break

        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_epoch_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                masks = torch.argmax(masks, dim=1)
                loss = total_loss(outputs, masks)
                running_val_loss += loss.item() * images.size(0)
                break

        val_epoch_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_epoch_loss:.4f}, "
            f"Val Loss: {val_epoch_loss:.4f}")
    return model

model = Segmentformer()
wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
class_weights = torch.tensor([wt0, wt1, wt2, wt3], dtype=torch.float32)
dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=4, ignore_index=None)
focal_loss = smp.losses.FocalLoss(mode='multiclass', ignore_index=None)
total_loss = lambda output, target: dice_loss(output, target) + focal_loss(output, target)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
# summary(model, (4, 128, 128, 128), 1)
model = train_model(model, optimizer=optimizer, total_loss=total_loss, train_loader=train_loader, val_loader=valid_loader)
model_save_path = '../ckpts/timesformer.hdf5'
torch.save(model.state_dict(), model_save_path)