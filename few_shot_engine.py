"""
Few-Shot Inference Engine
File: few_shot_engine.py
Chức năng: Phân loại ảnh sử dụng Prototypical Networks
"""
import torch
import torch.nn.functional as F
from typing import List, Dict
import numpy as np


class FewShotClassifier:
    """
    Few-Shot classifier sử dụng Prototypical Networks
    Nguyên lý: Tính prototype (vector đại diện) cho mỗi class,
    sau đó phân loại dựa trên khoảng cách Euclidean
    """
    
    def __init__(self, encoder, device='cpu'):
        """
        Khởi tạo classifier
        
        Args:
            encoder: Feature extractor đã được train (backbone của model)
            device: 'cuda' hoặc 'cpu'
        """
        self.encoder = encoder
        self.device = device
        self.encoder.to(device)
        self.encoder.eval()
        
        # Lưu trữ prototypes và class names
        self.prototypes = {}  # Dict: {class_name: prototype_vector}
        self.class_names = []  # List: [class1, class2, ...]
    
    def compute_prototype(self, features):
        """
        Tính prototype (vector trung bình) từ một tập features
        
        Args:
            features: Tensor [N, D] - N samples, D dimensions
        
        Returns:
            prototype: Tensor [D] - Vector đại diện cho class
        """
        return features.mean(dim=0)
    
    def build_support_set(self, support_images: Dict[str, List[torch.Tensor]]):
        """
        Xây dựng support set và tính prototypes cho tất cả classes
        
        Args:
            support_images: Dict {class_name: [img_tensor1, img_tensor2, ...]}
                          - Keys: Tên các class
                          - Values: List các image tensors [C, H, W]
        """
        print(f"\n🔨 Building support set...")
        self.prototypes = {}
        self.class_names = list(support_images.keys())
        
        with torch.no_grad():
            for class_name, images in support_images.items():
                # Stack images thành batch [N, C, H, W]
                imgs_tensor = torch.stack(images).to(self.device)
                
                # Extract features [N, D]
                features = self.encoder(imgs_tensor)
                
                # Tính prototype [D]
                prototype = self.compute_prototype(features)
                self.prototypes[class_name] = prototype
                
                print(f"    {class_name}: {len(images)} samples → prototype shape {prototype.shape}")
        
        print(f" Support set ready with {len(self.class_names)} classes\n")
    
    def classify(self, query_image: torch.Tensor, return_distances=False):
        """
        Phân loại một query image
        
        Args:
            query_image: Tensor [1, C, H, W] hoặc [C, H, W]
            return_distances: True nếu muốn trả về distances đến tất cả classes
        
        Returns:
            predicted_class: Tên class dự đoán (str)
            confidence: Độ tin cậy 0-1 (float)
            distances: Dict {class_name: distance} (nếu return_distances=True)
        """
        # Đảm bảo input có batch dimension
        if len(query_image.shape) == 3:
            query_image = query_image.unsqueeze(0)
        
        query_image = query_image.to(self.device)
        
        with torch.no_grad():
            # Extract feature từ query image [1, D] -> [D]
            query_feature = self.encoder(query_image).squeeze()
            
            # Tính khoảng cách Euclidean đến tất cả prototypes
            distances = {}
            for class_name, prototype in self.prototypes.items():
                # L2 distance = ||query - prototype||
                dist = torch.norm(query_feature - prototype, p=2).item()
                distances[class_name] = dist
            
            # Tìm class có khoảng cách nhỏ nhất (gần nhất)
            predicted_class = min(distances, key=distances.get)
            min_distance = distances[predicted_class]
            
            # Chuyển distances thành confidence scores
            # Sử dụng softmax trên negative distances (càng gần = càng cao)
            dist_tensor = torch.tensor(list(distances.values()))
            probs = F.softmax(-dist_tensor / 0.1, dim=0)  # temperature=0.1
            
            # Lấy confidence của predicted class
            confidence = probs[list(distances.keys()).index(predicted_class)].item()
        
        if return_distances:
            return predicted_class, confidence, distances
        return predicted_class, confidence
    
    def classify_batch(self, query_images: List[torch.Tensor]):
        """
        Phân loại nhiều images cùng lúc
        
        Args:
            query_images: List of image tensors [C, H, W]
        
        Returns:
            results: List of tuples (predicted_class, confidence)
        """
        results = []
        for img in query_images:
            pred, conf = self.classify(img)
            results.append((pred, conf))
        return results
    
    def add_new_class(self, class_name: str, sample_images: List[torch.Tensor]):
        """
        Thêm class mới vào support set (không cần training lại!)
        
        Args:
            class_name: Tên class mới
            sample_images: List of sample images [C, H, W]
        """
        if class_name in self.class_names:
            print(f" Class '{class_name}' đã tồn tại, sẽ cập nhật prototype")
        
        # Tính prototype cho class mới
        with torch.no_grad():
            imgs_tensor = torch.stack(sample_images).to(self.device)
            features = self.encoder(imgs_tensor)
            prototype = self.compute_prototype(features)
        
        # Lưu prototype
        self.prototypes[class_name] = prototype
        
        if class_name not in self.class_names:
            self.class_names.append(class_name)
        
        print(f" Added/Updated class '{class_name}' with {len(sample_images)} samples")
    
    def remove_class(self, class_name: str):
        """
        Xóa một class khỏi support set
        
        Args:
            class_name: Tên class cần xóa
        """
        if class_name in self.class_names:
            del self.prototypes[class_name]
            self.class_names.remove(class_name)
            print(f"Removed class '{class_name}'")
        else:
            print(f" Class '{class_name}' not found")
    
    def get_support_info(self):
        """
        Lấy thông tin về support set hiện tại
        
        Returns:
            Dict với thông tin về support set
        """
        return {
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'has_prototypes': len(self.prototypes) > 0
        }
    
    def save_prototypes(self, save_path):
        """
        Lưu prototypes ra file để sử dụng lại sau
        
        Args:
            save_path: Đường dẫn file .pth để lưu
        """
        torch.save({
            'prototypes': self.prototypes,
            'class_names': self.class_names
        }, save_path)
        print(f"Prototypes saved to {save_path}")
    
    def load_prototypes(self, load_path):
        """
        Load prototypes từ file đã lưu
        
        Args:
            load_path: Đường dẫn file .pth
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.prototypes = checkpoint['prototypes']
        self.class_names = checkpoint['class_names']
        print(f" Prototypes loaded from {load_path}")
        print(f"   Loaded {len(self.class_names)} classes: {self.class_names}")


def evaluate_fewshot_accuracy(classifier: FewShotClassifier, 
                               test_images: Dict[str, List[torch.Tensor]]):
    """
    Đánh giá accuracy của classifier trên test set
    
    Args:
        classifier: FewShotClassifier instance
        test_images: Dict {class_name: [img1, img2, ...]}
    
    Returns:
        accuracy: Overall accuracy (float)
        per_class_acc: Dict {class_name: accuracy}
    """
    correct = 0
    total = 0
    per_class_correct = {name: 0 for name in test_images.keys()}
    per_class_total = {name: 0 for name in test_images.keys()}
    
    print("\n Evaluating accuracy...")
    
    for true_class, images in test_images.items():
        for img in images:
            pred_class, conf = classifier.classify(img)
            
            if pred_class == true_class:
                correct += 1
                per_class_correct[true_class] += 1
            
            total += 1
            per_class_total[true_class] += 1
    
    # Tính overall accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Tính per-class accuracy
    per_class_acc = {
        name: per_class_correct[name] / per_class_total[name] 
        if per_class_total[name] > 0 else 0
        for name in test_images.keys()
    }
    
    # In kết quả
    print(f" Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("\nPer-class Accuracy:")
    for name, acc in per_class_acc.items():
        print(f"   {name}: {acc:.4f}")
    
    return accuracy, per_class_acc


if __name__ == "__main__":
    # Test code - chạy file này để kiểm tra
    print("Few-shot engine module loaded successfully")
    print("Usage:")
    print("   from few_shot_engine import FewShotClassifier")

    print("   classifier = FewShotClassifier(encoder, device)")
