"""
Model Loader for Few-Shot Plant Disease Classifier
File: model_loader.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor từ EfficientNet"""
    def __init__(self, backbone):
        super().__init__()
        self.level1 = backbone.features[:4]
        self.level2 = backbone.features[4:6]
        self.level3 = backbone.features[6:]
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.feat_dim_combined = 40 + 112 + 1280

    def forward(self, x):
        f1 = self.level1(x)
        f2 = self.level2(f1)
        f3 = self.level3(f2)
        f1_pool = torch.flatten(self.pool1(f1), 1)
        f2_pool = torch.flatten(self.pool2(f2), 1)
        f3_pool = torch.flatten(self.pool3(f3), 1)
        combined_feats = torch.cat([f1_pool, f2_pool, f3_pool], dim=1)
        return combined_feats


class SupConNet(nn.Module):
    """SupCon Network với projection head"""
    def __init__(self, backbone, feat_dim, proj_dim=128):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, proj_dim)
        )
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        feats = self.backbone(x)
        z = self.proj(feats)
        return F.normalize(z, dim=1)


def load_fewshot_model(model_path, device='cpu'):
    """
    Load mô hình few-shot đã train
    
    Args:
        model_path: Đường dẫn đến file .pth
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        model: Mô hình đã load
        encoder: Feature extractor (backbone)
    """
    print(f"🔄 Loading model from {model_path}...")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Khởi tạo architecture
    base_net = models.efficientnet_b0(pretrained=False)
    feature_extractor = MultiScaleFeatureExtractor(base_net)
    feat_dim = feature_extractor.feat_dim_combined
    
    # Tạo full model
    model = SupConNet(feature_extractor, feat_dim, proj_dim=128)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Chuyển sang eval mode
    model.eval()
    model.to(device)
    
    # Extract encoder (backbone)
    encoder = model.backbone
    encoder.eval()
    
    return model, encoder


def get_device():
    """Tự động chọn device phù hợp"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ GPU not available, using CPU")
    return device


class ModelConfig:
    """Configuration cho model"""
    IMG_SIZE = 224
    PROJ_DIM = 128
    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)


if __name__ == "__main__":
    # Test loading
    device = get_device()
    model_path = "models/fewshot_encoder_best.pth"
    
    try:
        model, encoder = load_fewshot_model(model_path, device)
        print(f"✅ Model architecture: {type(model).__name__}")
        print(f"✅ Encoder output dim: {encoder.feat_dim_combined}")
    except Exception as e:
        print(f"❌ Test failed: {e}")