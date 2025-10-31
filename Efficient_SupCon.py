
import os
import random
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DATASET_ROOT_TRAIN = '/kaggle/input/plantdoc-s/archive/train'  
DATASET_ROOT_TEST = '/kaggle/input/plantdoc-s/archive/test'    
IMG_SIZE = 224
TEMPERATURE = 0.07
PROJ_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 27  
SEED = 42
VAL_SPLIT = 0.2
NUM_WORKERS = 4
PATIENCE = 6  


BASE_BATCH = 64
BASE_EPOCHS_PRETRAIN = 50
BASE_EPOCHS_LINEAR = 30
LR_PRETRAIN = 1e-3
LR_LINEAR = 1e-3
WEIGHT_DECAY = 1e-4

# ---------------------------------------------------
random.seed(SEED)
torch.manual_seed(SEED)

# --------------------- Transforms ------------------
train_augment = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

class TwoCropTransform:
    """Create two random augmented crops of an image"""
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

# --------------------- Datasets --------------------
# Load datasets
full_train_plain = datasets.ImageFolder(DATASET_ROOT_TRAIN)  
num_samples = len(full_train_plain)

# adaptive hyperparams for small datasets
if num_samples < 5000:
    BATCH_SIZE = min(32, BASE_BATCH)
    EPOCHS_PRETRAIN = min(30, BASE_EPOCHS_PRETRAIN)
    EPOCHS_LINEAR = min(20, BASE_EPOCHS_LINEAR)
else:
    BATCH_SIZE = BASE_BATCH
    EPOCHS_PRETRAIN = BASE_EPOCHS_PRETRAIN
    EPOCHS_LINEAR = BASE_EPOCHS_LINEAR

print(f"Dataset has {num_samples} images. Using BATCH_SIZE={BATCH_SIZE}, EPOCHS_PRETRAIN={EPOCHS_PRETRAIN}, EPOCHS_LINEAR={EPOCHS_LINEAR}")

# build train/val splits
train_dataset = datasets.ImageFolder(DATASET_ROOT_TRAIN, transform=TwoCropTransform(train_augment))
train_size = int((1 - VAL_SPLIT) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# linear eval subsets (single-view)
train_linear_full = datasets.ImageFolder(DATASET_ROOT_TRAIN, transform=eval_transform)
train_linear_subset, val_linear_subset = random_split(train_linear_full, [train_size, val_size])

# test dataset
test_dataset = datasets.ImageFolder(DATASET_ROOT_TEST, transform=eval_transform)

# DataLoaders
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
val_loader_for_loss = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

train_linear_loader = DataLoader(train_linear_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_linear_loader = DataLoader(val_linear_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Dataset sizes: train={len(train_subset)}, val={len(val_subset)}, test={len(test_dataset)}")

# --------------------- (EfficientNet-B0) -----
base_net = models.efficientnet_b0(pretrained=True)
try:
    feat_dim = base_net.classifier[1].in_features
except Exception:
    feat_dim = 1280
base_net.classifier = nn.Identity()

class SupConNet(nn.Module):
    def __init__(self, backbone, feat_dim, proj_dim=128):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )
    def forward(self, x):
        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = torch.flatten(feats, 1)
        z = self.proj(feats)
        z = F.normalize(z, dim=1)
        return z

model = SupConNet(base_net, feat_dim, PROJ_DIM).to(DEVICE)

# --------------------- SupCon Loss -------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        if features.ndim != 3:
            raise ValueError('features must be [bsz, n_views, dim]')
        bsz = features.shape[0]
        n_views = features.shape[1]
        features = F.normalize(features, dim=2)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature

        logits = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if labels is None:
            mask = torch.eye(bsz, device=features.device)
        else:
            labels = labels.contiguous().view(-1,1)
            mask = torch.eq(labels, labels.T).float().to(features.device)
        mask = mask.repeat(n_views, n_views)

        logits_mask = (torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device))
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature) * mean_log_prob_pos
        loss = loss.view(n_views, bsz).mean()
        return loss

criterion = SupConLoss(temperature=TEMPERATURE)
optimizer = optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=WEIGHT_DECAY)
# Cosine annealing warm restarts for pretrain
scheduler_pre = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# --------------------- Pretraining loop ----------------
history = {'pretrain_train_loss': [], 'pretrain_val_loss': []}
print('==> Starting SupCon pretraining on device:', DEVICE)
for epoch in range(1, EPOCHS_PRETRAIN + 1):
    model.train()
    running_loss = 0.0
    for (images, labels) in train_loader:
        im1 = images[0].to(DEVICE)
        im2 = images[1].to(DEVICE)
        labels = labels.to(DEVICE)
        batch_size = labels.size(0)

        inputs = torch.cat([im1, im2], dim=0)
        feats = model(inputs)
        f1, f2 = torch.split(feats, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    history['pretrain_train_loss'].append(avg_loss)
    scheduler_pre.step()  # update LR scheduler

    # compute validation SupCon loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (v_images, v_labels) in val_loader_for_loss:
            v1 = v_images[0].to(DEVICE)
            v2 = v_images[1].to(DEVICE)
            vlabels = v_labels.to(DEVICE)
            b = vlabels.size(0)
            vin = torch.cat([v1, v2], dim=0)
            vfeats = model(vin)
            vf1, vf2 = torch.split(vfeats, [b, b], dim=0)
            vfeatures = torch.cat([vf1.unsqueeze(1), vf2.unsqueeze(1)], dim=1)
            vloss = criterion(vfeatures, vlabels)
            val_loss += vloss.item()
    val_loss = val_loss / len(val_loader_for_loss)
    history['pretrain_val_loss'].append(val_loss)

    print(f'Pretrain Epoch [{epoch}/{EPOCHS_PRETRAIN}] Train Loss: {avg_loss:.4f}  Val Loss: {val_loss:.4f}')

# save encoder weights
os.makedirs('/kaggle/working', exist_ok=True)
torch.save(model.state_dict(), '/kaggle/working/supcon_effnet_encoder.pth')

# plot pretraining losses
plt.figure()
plt.plot(history['pretrain_train_loss'], label='train_loss')
plt.plot(history['pretrain_val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SupCon Pretraining Loss')
plt.legend()
plt.grid(True)
plt.savefig('/kaggle/working/pretrain_loss.png')
plt.close()

# --------------------- Linear evaluation ----------------
print('==> Starting linear evaluation (freeze encoder)')
feature_extractor = model.backbone
feature_extractor.eval()
for p in feature_extractor.parameters():
    p.requires_grad = False

classifier = nn.Linear(feat_dim, NUM_CLASSES).to(DEVICE)
optimizer_cls = optim.Adam(classifier.parameters(), lr=LR_LINEAR, weight_decay=WEIGHT_DECAY)
# ReduceLROnPlateau based on val loss
scheduler_cls = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cls, mode='min', factor=0.5, patience=3, verbose=True)
criterion_cls = nn.CrossEntropyLoss()

history['linear_train_loss'] = []
history['linear_train_acc'] = []
history['linear_val_loss'] = []
history['linear_val_acc'] = []

best_val_acc = 0.0
best_model_wts = copy.deepcopy(classifier.state_dict())
patience_counter = 0

for epoch in range(1, EPOCHS_LINEAR + 1):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_linear_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.no_grad():
            feats = feature_extractor(images)
            if feats.ndim == 4:
                feats = torch.flatten(feats, 1)
        outputs = classifier(feats)
        loss = criterion_cls(outputs, labels)
        optimizer_cls.zero_grad()
        loss.backward()
        optimizer_cls.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = running_loss / len(train_linear_loader)

    # evaluate on validation
    classifier.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_linear_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            feats = feature_extractor(images)
            if feats.ndim == 4:
                feats = torch.flatten(feats, 1)
            outputs = classifier(feats)
            loss = criterion_cls(outputs, labels)
            val_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    val_loss = val_loss / len(val_linear_loader)

    history['linear_train_loss'].append(train_loss)
    history['linear_train_acc'].append(train_acc)
    history['linear_val_loss'].append(val_loss)
    history['linear_val_acc'].append(val_acc)

    print(f'Linear Epoch [{epoch}/{EPOCHS_LINEAR}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}  Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

    # scheduler step
    scheduler_cls.step(val_loss)

    # early stopping on val acc
    if val_acc > best_val_acc + 1e-5:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(classifier.state_dict())
        patience_counter = 0
        torch.save(classifier.state_dict(), '/kaggle/working/best_supcon_linear.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping triggered at epoch {epoch}')
            break

# load best weights
classifier.load_state_dict(best_model_wts)

# plot linear training history
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history['linear_train_loss'], label='train_loss')
plt.plot(history['linear_val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history['linear_train_acc'], label='train_acc')
plt.plot(history['linear_val_acc'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/kaggle/working/linear_training_history.png')
plt.close()

# final evaluation on test set
print('==> Evaluating on test set with best classifier')
classifier.eval()
correct = 0
total = 0
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        feats = feature_extractor(images)
        if feats.ndim == 4:
            feats = torch.flatten(feats, 1)
        outputs = classifier(feats)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

test_acc = correct / total
print(f'Test Accuracy: {test_acc:.4f}  Best Val Acc: {best_val_acc:.4f}')

# save final models
torch.save(model.state_dict(), '/kaggle/working/supcon_effnet_encoder_final.pth')
torch.save(classifier.state_dict(), '/kaggle/working/supcon_linear_final.pth')
print('Saved encoder and classifier to /kaggle/working')

# --------------------- Confusion Matrix & Report -------------------
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


class_names = test_dataset.classes

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(12,10))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'SupCon EfficientNet-B0 - Confusion Matrix (Test Acc={test_acc:.4f})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png')
plt.close()

# classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nClassification Report:\n", report)

with open('/kaggle/working/classification_report.txt', 'w') as f:
    f.write(report)

# test accuracy curve summary
plt.figure(figsize=(6,4))
plt.bar(['Val Acc (best)', 'Test Acc'], [best_val_acc, test_acc], color=['orange','green'])
plt.ylim(0,1)
plt.title('Validation vs Test Accuracy')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('/kaggle/working/val_test_accuracy_bar.png')
plt.close()

print("\n✅ Saved confusion_matrix.png, classification_report.txt, val_test_accuracy_bar.png to /kaggle/working")