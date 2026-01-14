"""
Utility functions cho Few-Shot App
File: utils.py
"""
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import os

# Transform cho inference
IMG_SIZE = 224
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])


def load_image(image_path_or_bytes, transform=None):
    """
    Load image từ đường dẫn hoặc bytes
    
    Args:
        image_path_or_bytes: Đường dẫn file hoặc bytes object
        transform: Transform để apply
    
    Returns:
        img_tensor: Tensor [C, H, W]
        img_pil: PIL Image (để hiển thị)
    """
    if isinstance(image_path_or_bytes, str):
        img_pil = Image.open(image_path_or_bytes).convert('RGB')
    else:
        img_pil = Image.open(image_path_or_bytes).convert('RGB')
    
    if transform is None:
        transform = inference_transform
    
    img_tensor = transform(img_pil)
    
    return img_tensor, img_pil


def load_images_from_folder(folder_path):
    import os
    tensors = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                # Giả sử load_image trả về (tensor, pil_image)
                res = load_image(img_path)
                
                # Kiểm tra nếu load_image trả về tuple (tensor, image)
                if isinstance(res, tuple):
                    tensor = res[0] # Lấy phần tử đầu tiên là Tensor
                else:
                    tensor = res
                
                tensors.append(tensor)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return tensors # Trả về list các Tensor duy nhất


def load_support_set_from_folders(base_path, transform=None, k_shot=5):
    """
    Load support set từ cấu trúc thư mục
    base_path/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg
    
    Args:
        base_path: Đường dẫn thư mục gốc
        transform: Transform để apply
        k_shot: Số lượng ảnh mỗi class (None = tất cả)
    
    Returns:
        support_images: Dict {class_name: [img_tensor1, ...]}
        class_names: List of class names
    """
    support_images = {}
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Folder not found: {base_path}")
    
    class_folders = [d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d))]
    
    for class_name in sorted(class_folders):
        class_path = os.path.join(base_path, class_name)
        images, _, _ = load_images_from_folder(class_path, transform, k_shot)
        
        if len(images) > 0:
            support_images[class_name] = images
            print(f"✅ Loaded {len(images)} images for class '{class_name}'")
        else:
            print(f"⚠️ No valid images found for class '{class_name}'")
    
    class_names = list(support_images.keys())
    
    return support_images, class_names


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Vẽ confusion matrix
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        class_names: List of class names
        save_path: Đường dẫn để lưu (None = không lưu)
    
    Returns:
        fig: Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_distance_distribution(distances_dict, predicted_class, save_path=None):
    """
    Vẽ biểu đồ phân bố khoảng cách
    
    Args:
        distances_dict: Dict {class_name: distance}
        predicted_class: Class được dự đoán
        save_path: Đường dẫn để lưu
    
    Returns:
        fig: Matplotlib figure
    """
    classes = list(distances_dict.keys())
    distances = list(distances_dict.values())
    
    colors = ['#2ecc71' if c == predicted_class else '#3498db' for c in classes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(classes, distances, color=colors, alpha=0.7)
    
    ax.set_xlabel('Distance to Prototype', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class', fontsize=12, fontweight='bold')
    ax.set_title('Distance Distribution', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Highlight predicted class
    for i, (bar, cls) in enumerate(zip(bars, classes)):
        if cls == predicted_class:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def display_sample_grid(images, titles=None, max_images=9, save_path=None):
    """
    Hiển thị grid của nhiều images
    
    Args:
        images: List of PIL Images
        titles: List of titles cho mỗi ảnh
        max_images: Số lượng ảnh tối đa hiển thị
        save_path: Đường dẫn để lưu
    
    Returns:
        fig: Matplotlib figure
    """
    n_images = min(len(images), max_images)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i in range(n_images):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i], fontsize=10, fontweight='bold')
    
    # Hide extra subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_temp_folder(folder_path='temp'):
    """Tạo thư mục tạm nếu chưa tồn tại"""
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def clean_temp_folder(folder_path='temp'):
    """Xóa tất cả files trong thư mục tạm"""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


def format_confidence(confidence):
    """Format confidence score thành text có màu"""
    if confidence >= 0.8:
        return f"🟢 **{confidence:.2%}** (High)"
    elif confidence >= 0.5:
        return f"🟡 **{confidence:.2%}** (Medium)"
    else:
        return f"🔴 **{confidence:.2%}** (Low)"


def get_color_by_confidence(confidence):
    """Trả về màu dựa trên confidence"""
    if confidence >= 0.8:
        return '#2ecc71'  # Green
    elif confidence >= 0.5:
        return '#f39c12'  # Orange
    else:
        return '#e74c3c'  # Red


if __name__ == "__main__":
    # Test utilities
    print("✅ Utilities module loaded successfully")
    print(f"   Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Normalize mean: {NORMALIZE_MEAN}")
    print(f"   Normalize std: {NORMALIZE_STD}")
