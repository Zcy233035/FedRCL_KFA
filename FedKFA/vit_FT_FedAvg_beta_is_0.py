"""
Federated Learning with Vision Transformer on CIFAR-100
Extreme Non-IID Scenario: 20 Clients with 1 Superclass Each (Beta → 0)

This implementation simulates extreme data heterogeneity where each of the 20 clients
receives data from exactly one of CIFAR-100's 20 superclasses, representing real-world
scenarios like specialized hospitals, regional preferences, or industry verticals.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import copy
import os
import json
from datetime import datetime
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# CIFAR-100 Superclass Mapping
CIFAR100_SUPERCLASSES = {
    'aquatic_mammals': [4, 30, 55, 72, 95],
    'fish': [1, 32, 67, 73, 91],
    'flowers': [54, 62, 70, 82, 92],
    'food_containers': [9, 10, 16, 28, 61],
    'fruit_and_vegetables': [0, 51, 53, 57, 83],
    'household_electrical_devices': [22, 39, 40, 86, 87],
    'household_furniture': [5, 20, 25, 84, 94],
    'insects': [6, 7, 14, 18, 24],
    'large_carnivores': [3, 42, 43, 88, 97],
    'large_man-made_outdoor_things': [12, 17, 37, 68, 76],
    'large_natural_outdoor_scenes': [23, 33, 49, 60, 71],
    'large_omnivores_and_herbivores': [15, 19, 21, 31, 38],
    'medium_mammals': [34, 63, 64, 66, 75],
    'non-insect_invertebrates': [26, 45, 77, 79, 99],
    'people': [2, 11, 35, 46, 98],
    'reptiles': [27, 29, 44, 78, 93],
    'small_mammals': [36, 50, 65, 74, 80],
    'trees': [47, 52, 56, 59, 96],
    'vehicles_1': [8, 13, 48, 58, 90],
    'vehicles_2': [41, 69, 81, 85, 89]
}

# 🚀 性能优化：创建全局transform pipeline，避免重复创建
import torchvision.transforms as transforms

# 全局transform pipeline，只创建一次
GLOBAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT标准输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
])

def transform_image(image, processor=None):
    """轻量级图像预处理函数"""
    # 🚀 直接使用全局transform，忽略processor参数
    return GLOBAL_TRANSFORM(image)

class FederatedClient:
    """Federated Learning Client for one superclass"""
    
    def __init__(self, client_id, superclass_name, superclass_classes, device=torch.device('cuda')):
        self.client_id = client_id
        self.superclass_name = superclass_name
        self.superclass_classes = superclass_classes
        self.device = device
        self.model = None
        self.processor = None
        self.train_loader = None
        self.optimizer = None
        
    def setup_model(self, global_model, processor):
        """Initialize client model from global model"""
        # 🚀 修复：只在CPU上保存模型，避免20个模型同时占用显存
        self.model = copy.deepcopy(global_model).cpu()  # 强制保存在CPU
        self.processor = processor
        # 初始化优化器但不立即创建，避免重复创建
        self.optimizer = None
        
    def setup_data(self, dataset, superclass_indices, processor, batch_size=32):
        """Setup client's data loader with pre-computed superclass indices"""
        # Create subset and transformed dataset
        client_subset = Subset(dataset, superclass_indices)
        client_transformed = TransformedCifar100(client_subset, processor)
        
        # 🚀 针对64GB配置的优化设置：num_workers=1 性价比最高
        # num_workers=0: 最稳定，但稍慢
        # num_workers=1: 性价比最佳，速度提升明显，内存增长有限
        # num_workers>1: 内存占用大增，但速度提升有限
        optimized_num_workers = 1  # 64GB配置下的最佳选择
        
        self.train_loader = DataLoader(
            client_transformed, 
            batch_size=batch_size,  # 保持原有batch_size，稳定为主
            shuffle=True,
            num_workers=optimized_num_workers,  # 轻度并行，性价比最高
            pin_memory=True,  # 64GB内存充足，可以启用
            persistent_workers=False  # 🚀 修复：禁用persistent_workers，避免内存累积
        )
        
        print(f"Client {self.client_id} ({self.superclass_name}): {len(superclass_indices)} samples")
        print(f"  Memory-Safe Config: batch_size={batch_size}, num_workers={optimized_num_workers}, persistent_workers=False")
        
    def local_train(self, epochs=1, lr=2e-5):
        """Perform local training"""
        print(f"    Training Client {self.client_id} ({self.superclass_name})...")
        
        # 类型检查：确保train_loader已初始化
        if self.train_loader is None:
            raise ValueError(f"Client {self.client_id}: train_loader not initialized")
        
        # 🚀 修复：训练前将模型移到GPU
        self.model = self.model.to(self.device)  # type: ignore
        self.model.train()
        
        # 只在第一次或需要时创建优化器
        if self.optimizer is None:
            if self.model is None:
                raise ValueError(f"Client {self.client_id}: model not initialized")
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        total_loss = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            # 添加批次级别的进度条
            batch_pbar = tqdm(self.train_loader, desc=f"      Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch_idx, batch in enumerate(batch_pbar):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                total_samples += labels.size(0)
                
                # 更新进度条显示当前loss
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            total_loss += epoch_loss
        
        # 🚀 修复：训练完成后立即将模型移回CPU，释放GPU显存
        self.model = self.model.cpu()
        torch.cuda.empty_cache()  # 清理GPU缓存
        
        # 🚀 新增：强制内存清理，解决累积问题
        import gc
        gc.collect()  # 强制垃圾回收
        
        # 清理可能的临时变量
        del batch_pbar
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待GPU操作完成
            torch.cuda.empty_cache()  # 再次清理GPU缓存
            
        avg_loss = total_loss / (epochs * len(self.train_loader))
        print(f"    Client {self.client_id} finished: avg_loss={avg_loss:.4f}")
        return avg_loss, total_samples
    
    def get_model_parameters(self):
        """Get model parameters for aggregation"""
        if self.model is None:
            raise ValueError(f"Client {self.client_id}: model not initialized")
        return {name: param.cpu().clone() for name, param in self.model.named_parameters()}
    
    def cleanup_resources(self):
        """清理客户端资源，防止内存泄漏"""
        # 🚀 修复：不完全删除train_loader，只清理内部资源
        # 清理DataLoader的内部缓存和迭代器，但保留DataLoader对象
        if hasattr(self, 'train_loader') and self.train_loader is not None:
            try:
                # 清理可能存在的迭代器
                if hasattr(self.train_loader, '_iterator'):
                    del self.train_loader._iterator
                # 不删除train_loader本身，因为后续轮次还需要使用
            except:
                pass
        
        # 清理优化器（下次训练时会重新创建）
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class FederatedServer:
    """Federated Learning Server implementing FedAvg"""
    
    def __init__(self, device=torch.device('cuda')):
        self.device = device
        self.global_model = None
        self.processor = None
        self.clients = []
        self.test_loader = None
        
        # Tracking metrics
        self.round_history = {
            'round': [],
            'global_accuracy': [],
            'superclass_accuracies': [],
            'avg_client_loss': [],
            'num_participating_clients': []
        }
        
    def setup_global_model(self):
        """Initialize global ViT model"""
        try:
            self.processor = ViTImageProcessor.from_pretrained(
                'google/vit-base-patch16-224',
                do_resize=True,
                size=224,
                do_rescale=False  # 图像已经在[0,1]范围，不需要重新rescale
            )
            
            self.global_model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=100,  # CIFAR-100 has 100 classes
                ignore_mismatched_sizes=True
            )
            if isinstance(self.device, str):
                self.global_model = self.global_model.to(torch.device(self.device))  # type: ignore
            else:
                self.global_model = self.global_model.to(self.device)  # type: ignore
            
            print("Global ViT model initialized for CIFAR-100")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize global model: {e}")
            
        # 🚀 安全检查：确保初始化成功
        if self.global_model is None or self.processor is None:
            raise RuntimeError("Global model or processor initialization failed")
        
    def setup_test_data(self, batch_size=64):
        """Setup global test set"""
        test_dataset = CIFAR100(root='./data', train=False, download=True)
        test_transformed = TransformedCifar100(test_dataset, self.processor)
        
        # 🚀 测试时可以稍大batch_size，因为无需梯度计算
        # 但仍然保守，避免显存问题
        safe_batch_size = 64  # 测试时适中的batch_size
        
        self.test_loader = DataLoader(
            test_transformed, 
            batch_size=safe_batch_size, 
            shuffle=False, 
            num_workers=1,  # 测试时也使用轻度并行
            pin_memory=True,
            persistent_workers=False  # 🚀 修复：测试时也禁用persistent_workers
        )
        print(f"Test set: {len(test_dataset)} samples (batch_size={safe_batch_size})")
        
    def register_client(self, client):
        """Register a client"""
        self.clients.append(client)
        
    def distribute_model(self):
        """Send global model to all clients with memory optimization"""
        for client in self.clients:
            # 🚀 内存优化：先清理旧模型再分发新模型
            if hasattr(client, 'model') and client.model is not None:
                del client.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            client.setup_model(self.global_model, self.processor)
            
    def aggregate_models(self, client_updates):
        """Aggregate client model updates using FedAvg with memory optimization"""
        if self.global_model is None:
            raise ValueError("Global model not initialized")
            
        # 🚀 安全检查：确保有客户端更新
        if not client_updates:
            raise ValueError("No client updates received for aggregation")
            
        # Calculate total samples for weighted averaging
        total_samples = sum(samples for _, samples in client_updates)
        
        # 🚀 安全检查：确保有训练样本
        if total_samples == 0:
            raise ValueError("Total training samples is zero - all clients failed")
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # 🚀 内存优化：流式处理客户端更新，减少峰值内存
        for i, (client_params, num_samples) in enumerate(client_updates):
            weight = num_samples / total_samples
            
            for name, param in client_params.items():
                if name not in aggregated_params:
                    aggregated_params[name] = weight * param.clone()
                else:
                    aggregated_params[name] += weight * param
            
            # 立即释放已处理的客户端参数，减少内存累积
            del client_params
            
            # 每处理5个客户端就清理一次内存
            if (i + 1) % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Update global model
        global_dict = self.global_model.state_dict()
        for name, param in aggregated_params.items():
            if name in global_dict:
                global_dict[name] = param
        
        self.global_model.load_state_dict(global_dict)
        
        # 清理聚合参数
        del aggregated_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def evaluate_global_model(self):
        """Evaluate global model on test set"""
        if self.global_model is None:
            raise ValueError("Global model not initialized")
        if self.test_loader is None:
            raise ValueError("Test loader not initialized")
            
        self.global_model.eval()
        correct = 0
        total = 0
        superclass_correct = defaultdict(int)
        superclass_total = defaultdict(int)
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.global_model(pixel_values=pixel_values)
                _, predicted = torch.max(outputs.logits, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Track superclass performance
                for i, label in enumerate(labels):
                    for superclass_name, classes in CIFAR100_SUPERCLASSES.items():
                        if label.item() in classes:
                            superclass_total[superclass_name] += 1
                            if predicted[i] == label:
                                superclass_correct[superclass_name] += 1
                            break
        
        global_accuracy = 100 * correct / total
        
        # Calculate superclass accuracies
        superclass_accuracies = {}
        for superclass_name in CIFAR100_SUPERCLASSES.keys():
            if superclass_total[superclass_name] > 0:
                acc = 100 * superclass_correct[superclass_name] / superclass_total[superclass_name]
                superclass_accuracies[superclass_name] = acc
            else:
                superclass_accuracies[superclass_name] = 0.0
                
        return global_accuracy, superclass_accuracies
    
    def federated_round(self, round_num, local_epochs=1):
        """Execute one round of federated learning"""
        print(f"\n=== Federated Round {round_num} ===")
        
        # Distribute global model to clients
        self.distribute_model()
        
        # Client local training
        client_updates = []
        total_loss = 0
        failed_clients = 0
        
        for client in tqdm(self.clients, desc="Client Training"):
            try:
                loss, num_samples = client.local_train(epochs=local_epochs)
                client_params = client.get_model_parameters()
                client_updates.append((client_params, num_samples))
                total_loss += loss
                
                # 🚀 新增：每个客户端训练完成后立即清理资源
                client.cleanup_resources()
            except Exception as e:
                print(f"⚠️  Client {client.client_id} failed: {e}")
                failed_clients += 1
                # 继续训练其他客户端，不中断整个过程
                continue
        
        # 🚀 安全检查：确保至少有一些客户端成功
        if len(client_updates) == 0:
            raise RuntimeError(f"All {len(self.clients)} clients failed in round {round_num}")
        
        if failed_clients > 0:
            print(f"⚠️  {failed_clients} clients failed, continuing with {len(client_updates)} successful clients")
            
        avg_client_loss = total_loss / len(client_updates)  # 使用成功的客户端数量
        
        # Server aggregation
        self.aggregate_models(client_updates)
        
        # Global evaluation
        global_accuracy, superclass_accuracies = self.evaluate_global_model()
        
        # Record metrics
        self.round_history['round'].append(round_num)
        self.round_history['global_accuracy'].append(global_accuracy)
        self.round_history['superclass_accuracies'].append(superclass_accuracies)
        self.round_history['avg_client_loss'].append(avg_client_loss)
        self.round_history['num_participating_clients'].append(len(self.clients))
        
        print(f"Round {round_num} Results:")
        print(f"  Global Accuracy: {global_accuracy:.2f}%")
        print(f"  Average Client Loss: {avg_client_loss:.4f}")
        print(f"  Participating Clients: {len(self.clients)}")
        
        return global_accuracy, superclass_accuracies, avg_client_loss

class TransformedCifar100(torch.utils.data.Dataset):
    """预处理后的CIFAR100数据集，避免训练时重复处理"""
    def __init__(self, dataset, processor):
        self.dataset = dataset
        # 🚀 性能优化：预先存储处理器参数，避免重复调用
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self.dataset, 'indices'):  # Subset
            image, label = self.dataset.dataset[self.dataset.indices[idx]]
        else:
            image, label = self.dataset[idx]
        
        # 🚀 优化：使用轻量级transform替代重复的processor调用
        pixel_values = transform_image(image, self.processor)
        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}

def setup_federated_data():
    """Setup federated data distribution with optimized indexing"""
    # 基础数据集，不做预处理
    train_dataset = CIFAR100(root='./data', train=True, download=True)
    test_dataset = CIFAR100(root='./data', train=False, download=True)
    print(f"Total training samples: {len(train_dataset)}")
    
    # 🚀 预先构建超类到样本索引的映射，只遍历一次！
    print("Building superclass index mapping...")
    superclass_to_indices = {name: [] for name in CIFAR100_SUPERCLASSES.keys()}
    
    for idx, (_, label) in enumerate(train_dataset):  # type: ignore
        for superclass_name, classes in CIFAR100_SUPERCLASSES.items():
            if label in classes:
                superclass_to_indices[superclass_name].append(idx)
                break
    
    # 验证数据分布
    for name, indices in superclass_to_indices.items():
        print(f"Superclass {name}: {len(indices)} samples")
    
    return train_dataset, superclass_to_indices

def create_visualizations(server, save_dir):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Federated Learning Results: 20 Clients, 1 Superclass Each (β→0)', fontsize=16)
    
    # 1. Global Accuracy Over Rounds
    axes[0, 0].plot(server.round_history['round'], server.round_history['global_accuracy'], 
                   'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Global Test Accuracy')
    axes[0, 0].set_xlabel('Federated Round')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Average Client Loss
    axes[0, 1].plot(server.round_history['round'], server.round_history['avg_client_loss'], 
                   'r-s', linewidth=2, markersize=6)
    axes[0, 1].set_title('Average Client Training Loss')
    axes[0, 1].set_xlabel('Federated Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Final Superclass Performance Heatmap
    if server.round_history['superclass_accuracies']:
        final_superclass_acc = server.round_history['superclass_accuracies'][-1]
        superclass_names = list(final_superclass_acc.keys())
        accuracies = list(final_superclass_acc.values())
        
        # Create heatmap data
        acc_matrix = np.array(accuracies).reshape(1, -1)
        im = axes[0, 2].imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        axes[0, 2].set_title('Final Superclass Accuracies')
        axes[0, 2].set_xticks(range(len(superclass_names)))
        axes[0, 2].set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                                  for name in superclass_names], rotation=45, ha='right')
        axes[0, 2].set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[0, 2])
        cbar.set_label('Accuracy (%)')
    
    # 4. Superclass Performance Evolution
    if len(server.round_history['superclass_accuracies']) > 1:
        selected_superclasses = ['vehicles_1', 'people', 'flowers', 'large_carnivores', 'fish']
        for superclass in selected_superclasses:
            if superclass in CIFAR100_SUPERCLASSES:
                accs = [round_acc.get(superclass, 0) for round_acc in server.round_history['superclass_accuracies']]
                axes[1, 0].plot(server.round_history['round'], accs, 
                              'o-', linewidth=2, label=superclass, markersize=4)
        
        axes[1, 0].set_title('Selected Superclass Performance Evolution')
        axes[1, 0].set_xlabel('Federated Round')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Data Distribution Visualization
    superclass_sizes = [len(classes) for classes in CIFAR100_SUPERCLASSES.values()]
    axes[1, 1].bar(range(len(CIFAR100_SUPERCLASSES)), superclass_sizes)
    axes[1, 1].set_title('Classes per Superclass (All Equal: 5)')
    axes[1, 1].set_xlabel('Superclass Index')
    axes[1, 1].set_ylabel('Number of Classes')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary Statistics
    if server.round_history['global_accuracy']:
        final_acc = server.round_history['global_accuracy'][-1]
        max_acc = max(server.round_history['global_accuracy'])
        min_acc = min(server.round_history['global_accuracy'])
        
        summary_text = f"""
Final Results Summary:
• Total Rounds: {len(server.round_history['round'])}
• Final Global Accuracy: {final_acc:.2f}%
• Best Global Accuracy: {max_acc:.2f}%
• Worst Global Accuracy: {min_acc:.2f}%
• Number of Clients: 20
• Data Distribution: Extreme Non-IID
• Each Client: 1 Superclass (5 classes)
• Algorithm: FedAvg
• Model: ViT-base-patch16-224
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'federated_learning_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def save_results(server, save_dir):
    """Save all results and models"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save global model
    torch.save(server.global_model.state_dict(), 
               os.path.join(save_dir, 'global_model.pth'))
    server.processor.save_pretrained(os.path.join(save_dir, 'processor'))
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        history_json = {}
        for key, value in server.round_history.items():
            if key == 'superclass_accuracies':
                history_json[key] = value  # List of dicts
            else:
                history_json[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
        json.dump(history_json, f, indent=2)
    
    # Save experiment metadata
    metadata = {
        'experiment_type': 'Federated Learning - Extreme Non-IID (Beta → 0)',
        'model': 'ViT-base-patch16-224',
        'dataset': 'CIFAR-100',
        'num_clients': len(server.clients),
        'data_distribution': 'Each client gets 1 superclass (5 classes)',
        'algorithm': 'FedAvg',
        'num_rounds': len(server.round_history['round']),
        'final_global_accuracy': float(server.round_history['global_accuracy'][-1]) if server.round_history['global_accuracy'] else None,
        'timestamp': datetime.now().isoformat(),
        'superclass_mapping': CIFAR100_SUPERCLASSES
    }
    
    with open(os.path.join(save_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")
    print(f"- Global model: global_model.pth")
    print(f"- Processor: processor/")
    print(f"- Training history: training_history.json")
    print(f"- Metadata: experiment_metadata.json")
    print(f"- Visualizations: federated_learning_results.png")

def main(resume_from_checkpoint=None):
    """Main federated learning execution"""
    print("=" * 60)
    print("Federated Learning with ViT on CIFAR-100")
    print("Extreme Non-IID: 20 Clients, 1 Superclass Each (β→0)")
    if resume_from_checkpoint:
        print(f"📂 Resuming from checkpoint: {resume_from_checkpoint}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 🚀 添加GPU检查
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Current GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")
    else:
        print("⚠️  CUDA not available, will use CPU (very slow)")
    
    # Initialize server
    server = FederatedServer(device=device)
    server.setup_global_model()
    server.setup_test_data()
    
    # 🚀 从checkpoint恢复训练状态
    start_round = 1
    if resume_from_checkpoint:
        try:
            print(f"📂 Loading checkpoint from {resume_from_checkpoint}")
            
            # 调试：检查文件是否存在
            model_path = os.path.join(resume_from_checkpoint, 'global_model.pth')
            history_path = os.path.join(resume_from_checkpoint, 'training_history.json')
            print(f"🔍 Model file exists: {os.path.exists(model_path)}")
            print(f"🔍 History file exists: {os.path.exists(history_path)}")
            
            # 加载训练历史（先加载历史再加载模型）
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                server.round_history = history
                start_round = len(history['round']) + 1
                print(f"✅ Training history loaded, resuming from round {start_round}")
                print(f"   Previous best accuracy: {max(history['global_accuracy']):.2f}%")
            else:
                print("❌ Training history file not found!")
            
            # 加载模型权重
            if os.path.exists(model_path) and server.global_model is not None:
                state_dict = torch.load(model_path, map_location=device)
                server.global_model.load_state_dict(state_dict)
                print("✅ Global model loaded successfully")
            else:
                print("❌ Model file not found or server.global_model is None!")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("Starting fresh training...")
            start_round = 1
    
    # Setup federated data with optimized indexing
    train_dataset, superclass_to_indices = setup_federated_data()
    
    # Create 20 clients, one for each superclass
    print("\nCreating 20 federated clients...")
    for client_id, (superclass_name, superclass_classes) in enumerate(CIFAR100_SUPERCLASSES.items()):
        client = FederatedClient(
            client_id=client_id,
            superclass_name=superclass_name,
            superclass_classes=superclass_classes,
            device=device
        )
        
        # Setup client's data using pre-computed indices
        client.setup_data(train_dataset, superclass_to_indices[superclass_name], server.processor, batch_size=32)
        server.register_client(client)
    
    print(f"\nFederated setup complete:")
    print(f"- Server: 1 global ViT model")
    print(f"- Clients: {len(server.clients)}")
    print(f"- Data distribution: Extreme Non-IID (1 superclass per client)")
    
    # Run federated learning
    num_rounds = 25  # 🚀 看看ViT的真实上限！
    local_epochs = 3
    
    print(f"\nStarting federated learning...")
    print(f"- Rounds: {num_rounds}")
    print(f"- Local epochs per round: {local_epochs}")
    
    # 🚀 移除Ctrl+C中断机制，避免误触发
    try:
        for round_num in range(start_round, num_rounds + 1):
            print(f"\n🚀 Starting Round {round_num}/{num_rounds}")
            global_acc, superclass_accs, avg_loss = server.federated_round(
                round_num, local_epochs=local_epochs
            )
            
            # Print top and bottom performing superclasses
            sorted_superclasses = sorted(superclass_accs.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 3 superclasses: {sorted_superclasses[:3]}")
            print(f"  Bottom 3 superclasses: {sorted_superclasses[-3:]}")
            
            # 🚀 每轮后保存中间结果，防止长时间训练丢失
            if round_num % 2 == 0 or round_num == num_rounds:  # 每2轮或最后一轮保存
                temp_save_dir = f"federated_results_round_{round_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    save_results(server, temp_save_dir)
                    print(f"  ✅ Intermediate results saved to: {temp_save_dir}")
                except Exception as e:
                    print(f"  ⚠️ Failed to save intermediate results: {e}")
                    
    except Exception as e:
        print(f"\n❌ Training failed at round {round_num}: {e}")
        print("Saving progress before exit...")
        error_save_dir = f"federated_results_error_round_{round_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            save_results(server, error_save_dir)
            print(f"Progress saved to: {error_save_dir}")
        except:
            print("❌ Failed to save progress")
        raise  # 重新抛出错误以便调试
    
    # Final evaluation and visualization
    print(f"\nFederated learning completed!")
    final_global_acc = server.round_history['global_accuracy'][-1]
    print(f"Final global accuracy: {final_global_acc:.2f}%")
    
    # Save results
    save_dir = f"federated_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_results(server, save_dir)
    
    # Create visualizations
    create_visualizations(server, save_dir)
    
    print("\nExperiment completed successfully!")
    return server

if __name__ == "__main__":
    # 🚀 使用示例：
    # 从头开始训练：
    # server = main()
    
    # 从checkpoint继续训练（把路径改成你的checkpoint文件夹）：
    # server = main(resume_from_checkpoint="federated_results_round_10_20250617_XXXXXX")
    
    # 🚀 从第8轮继续训练到第20轮
    server = main(resume_from_checkpoint="federated_results_round_8_20250617_224342")
