# ============================================================
# 1. IMPORTS & CONFIG
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import copy # Cần cho Early Stopping

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

BATCH_SIZE = 32
EPOCHS_PRETRAIN = 40
EPOCHS_LINEAR = 20
IMG_SIZE = 224
TEMPERATURE = 0.07
LR_PRETRAIN = 3e-4
LR_LINEAR = 1e-5
VAL_SPLIT = 0.15
PATIENCE = 7 # <-- THÊM: Số epoch không cải thiện trước khi dừng

OUT_DIR = "/kaggle/working/ResNet50"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2. DATASET
# ============================================================
ROOT = "/kaggle/input/plantdoc-s/archive"  # thay bằng thư mục dataset của bạn
train_dir = os.path.join(ROOT, "train")
test_dir = os.path.join(ROOT, "test")

transform_base = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_supcon = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class SupConDataset(Dataset):
    def __init__(self, subset, base_transform, supcon_transform):
        self.subset = subset
        self.base_transform = base_transform
        self.supcon_transform = supcon_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Lấy chỉ số ảnh trong dataset gốc
        real_idx = self.subset.indices[idx]
        # Lấy đường dẫn và nhãn từ dataset gốc
        path, label = self.subset.dataset.samples[real_idx]
        # Load ảnh
        img = self.subset.dataset.loader(path)
        # Tạo hai augmentations khác nhau cho SupCon
        xi = self.supcon_transform(img)
        xj = self.supcon_transform(img)
        return xi, xj, label

# load full dataset
full_train = datasets.ImageFolder(train_dir, transform=transform_base)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_base)

# split train/val
val_len = int(len(full_train) * VAL_SPLIT)
train_len = len(full_train) - val_len
train_subset, val_subset = random_split(full_train, [train_len, val_len])

supcon_train = SupConDataset(train_subset, transform_base, transform_supcon)
# THÊM: SupCon Val Dataset để theo dõi Val Loss (dù không dùng cho Early Stop)
supcon_val = SupConDataset(val_subset, transform_base, transform_supcon) 

train_loader_supcon = DataLoader(supcon_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True) # drop_last=True cần cho supcon
val_loader_supcon = DataLoader(supcon_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True) # drop_last=True cần cho supcon
train_loader_linear = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader_linear = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================
# 3. MODEL - ResNet50 Encoder
# ============================================================
class SupConResNet50(nn.Module):
    def __init__(self, pretrained=True, proj_dim=128):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, proj_dim)
        )

    def forward(self, x):
        feats = self.encoder(x)
        z = F.normalize(self.projection_head(feats), dim=1)
        return z, feats

model = SupConResNet50().to(DEVICE)

# ---- SupCon LOSS

def supcon_loss(features, labels, temperature=0.07):
    device = features.device
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    contrast = torch.matmul(features, features.T) / temperature
    logits_max, _ = torch.max(contrast, dim=1, keepdim=True)
    logits = contrast - logits_max.detach()
    exp_logits = torch.exp(logits)
    mask_self = torch.eye(mask.shape[0], dtype=torch.bool).to(device)
    mask = mask * ~mask_self
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss


#---- PRETRAIN CONTRASTIVE

optimizer = optim.Adam(model.parameters(), lr=LR_PRETRAIN)
#  Cosine Annealing cho Pretraining
scheduler_pre = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) 
train_losses = []
val_losses_supcon = []

for epoch in range(EPOCHS_PRETRAIN):
    model.train()
    total_loss = 0
    for xi, xj, labels in train_loader_supcon:
        xi, xj, labels = xi.to(DEVICE), xj.to(DEVICE), labels.to(DEVICE)
        zi, _ = model(xi)
        zj, _ = model(xj)
        z = torch.cat([zi, zj], dim=0)
        y = torch.cat([labels, labels], dim=0)
        loss = supcon_loss(z, y, TEMPERATURE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        scheduler_pre.step() # Cập nhật scheduler sau mỗi batch

    avg_loss = total_loss / len(train_loader_supcon)
    train_losses.append(avg_loss)
    
    # Validation SupCon Loss
    model.eval()
    val_loss_supcon = 0.0
    with torch.no_grad():
        for xi, xj, labels in val_loader_supcon:
            xi, xj, labels = xi.to(DEVICE), xj.to(DEVICE), labels.to(DEVICE)
            zi, _ = model(xi)
            zj, _ = model(xj)
            z = torch.cat([zi, zj], dim=0)
            y = torch.cat([labels, labels], dim=0)
            loss = supcon_loss(z, y, TEMPERATURE)
            val_loss_supcon += loss.item()
    val_loss_supcon /= len(val_loader_supcon)
    val_losses_supcon.append(val_loss_supcon)

    print(f"[Pretrain] Epoch {epoch+1}/{EPOCHS_PRETRAIN} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss_supcon:.4f} (LR: {optimizer.param_groups[0]['lr']:.6f})")


plt.figure()
plt.plot(train_losses, label="SupCon Train Loss")
plt.plot(val_losses_supcon, label="SupCon Val Loss") # Plot Val Loss
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Supervised Contrastive Pretraining (ResNet50)")
plt.savefig(f"{OUT_DIR}/pretrain_loss.png")
plt.close()

# save encoder 
torch.save(model.state_dict(), f"{OUT_DIR}/supcon_resnet50_encoder.pth")
print("✅ Pretrained ResNet50 encoder saved.")


# LINEAR CLASSIFIER TRAINING

feature_extractor = model.encoder
feature_extractor.eval()
for param in feature_extractor.parameters():
    param.requires_grad = False #freeze

num_classes = len(full_train.classes)
classifier = nn.Linear(model.feature_dim, num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=LR_LINEAR)

scheduler_cls = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)


best_val_acc = 0.0
best_model_wts = copy.deepcopy(classifier.state_dict()) 
patience_counter = 0 
train_hist, val_hist = [], []

for epoch in range(EPOCHS_LINEAR):
   
    if patience_counter >= PATIENCE:
        print(f"🛑 Early stopping triggered at epoch {epoch}. Patience: {PATIENCE}")
        break

    classifier.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader_linear:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            feats = feature_extractor(images)
        outputs = classifier(feats)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    train_loss = total_loss / len(train_loader_linear)

    # validation
    classifier.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader_linear:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            feats = feature_extractor(images)
            outputs = classifier(feats)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    val_loss /= len(val_loader_linear)

    train_hist.append(train_acc)
    val_hist.append(val_acc)
    print(f"[Linear] Epoch {epoch+1}/{EPOCHS_LINEAR}  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f} (LR: {optimizer.param_groups[0]['lr']:.6f})")


    scheduler_cls.step(val_acc) 

   
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(classifier.state_dict())
        torch.save(classifier.state_dict(), f"{OUT_DIR}/best_linear_resnet50.pth")
        patience_counter = 0 # Reset bộ đếm
    else:
        patience_counter += 1 # Tăng bộ đếm

# Load best weights
classifier.load_state_dict(best_model_wts)

plt.figure()
plt.plot(train_hist, label="Train Acc")
plt.plot(val_hist, label="Val Acc")
plt.legend()
plt.title("Linear Evaluation (ResNet50 SupCon)")
plt.savefig(f"{OUT_DIR}/linear_training_history.png")
plt.close()


print("==> Evaluating on test set...")
# Khởi tạo lại với trạng thái tốt nhất đã lưu
classifier.load_state_dict(torch.load(f"{OUT_DIR}/best_linear_resnet50.pth")) 
classifier.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        feats = feature_extractor(images)
        outputs = classifier(feats)
        _, preds = outputs.max(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

test_acc = np.mean(np.array(y_true) == np.array(y_pred))
print(f"✅ Test Accuracy: {test_acc:.4f}  |  Best Val Acc: {best_val_acc:.4f}")

# confusion matrix
class_names = test_dataset.classes
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(12,10))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"ResNet50 SupCon - Confusion Matrix (Test Acc={test_acc:.4f})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_resnet50.png")
plt.close()

# classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
with open(f"{OUT_DIR}/classification_report_resnet50.txt", "w") as f:
    f.write(report)
print("\nClassification Report:\n", report)

print(f"\n🎉 All artifacts saved in {OUT_DIR}")