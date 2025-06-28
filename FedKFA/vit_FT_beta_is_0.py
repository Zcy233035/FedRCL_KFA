import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# --- 配置参数 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

MODEL_ID = "google/vit-base-patch16-224"
DATA_DIR = "./data"
SAVE_DIR = "./saved_models"
TARGET_SUPERCLASS = "vehicles_1"  # 选择目标超类
NUM_EPOCHS = 3  # 增加训练轮次以看到更明显的效果
BATCH_SIZE = 32
LEARNING_RATE = 2e-5

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"🎯 实验设置：只使用超类 '{TARGET_SUPERCLASS}' 的数据进行微调")
print(f"📊 这模拟了联邦学习中 beta→0 的极端数据偏斜情况")

# --- 1. 数据准备 ---
processor = ViTImageProcessor.from_pretrained(MODEL_ID)

def transform(examples):
    inputs = processor(images=examples, return_tensors="pt")
    return inputs['pixel_values']

# 加载完整数据集
train_dataset = CIFAR100(root=DATA_DIR, train=True, download=True)
test_dataset = CIFAR100(root=DATA_DIR, train=False, download=True)

# CIFAR-100 超类定义
superclass_mapping = {
    "aquatic_mammals": [4, 30, 55, 72, 95],  # beaver, dolphin, otter, seal, whale
    "fish": [1, 32, 67, 73, 91],  # aquarium_fish, flatfish, ray, shark, trout
    "flowers": [54, 62, 70, 82, 92],  # orchid, poppy, rose, sunflower, tulip
    "food_containers": [9, 10, 16, 28, 61],  # bottle, bowl, can, cup, plate
    "fruit_and_vegetables": [0, 51, 53, 57, 83],  # apple, mushroom, orange, pear, sweet_pepper
    "household_electrical_devices": [22, 39, 40, 86, 87],  # clock, keyboard, lamp, telephone, television
    "household_furniture": [5, 20, 25, 84, 94],  # bed, chair, couch, table, wardrobe
    "insects": [6, 7, 14, 18, 24],  # bee, beetle, butterfly, caterpillar, cockroach
    "large_carnivores": [3, 42, 43, 88, 97],  # bear, leopard, lion, tiger, wolf
    "large_man_made_outdoor_things": [12, 17, 37, 68, 76],  # bridge, castle, house, road, skyscraper
    "large_natural_outdoor_scenes": [23, 33, 49, 60, 71],  # cloud, forest, mountain, plain, sea
    "large_omnivores_and_herbivores": [15, 19, 21, 31, 38],  # camel, cattle, chimpanzee, elephant, kangaroo
    "medium_mammals": [34, 63, 64, 66, 75],  # fox, porcupine, possum, raccoon, skunk
    "non_insect_invertebrates": [26, 45, 77, 79, 99],  # crab, lobster, snail, spider, worm
    "people": [2, 11, 35, 46, 98],  # baby, boy, girl, man, woman
    "reptiles": [27, 29, 44, 78, 93],  # crocodile, dinosaur, lizard, snake, turtle
    "small_mammals": [36, 50, 65, 74, 80],  # hamster, mouse, rabbit, shrew, squirrel
    "trees": [47, 52, 56, 59, 96],  # maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
    "vehicles_1": [8, 13, 48, 58, 90],  # bicycle, bus, motorcycle, pickup_truck, train
    "vehicles_2": [41, 69, 81, 85, 89]  # lawn_mower, rocket, streetcar, tank, tractor
}

# CIFAR-100类别名称
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# 获取目标超类的所有类别
target_classes = superclass_mapping[TARGET_SUPERCLASS]
target_class_names = [class_names[i] for i in target_classes]

print(f"🏷️  目标超类: {TARGET_SUPERCLASS}")
print(f"📝 包含类别: {', '.join(target_class_names)}")
print(f"🔢 类别索引: {target_classes}")

# 创建只包含目标超类的训练集
target_indices = [i for i, (_, label) in enumerate(train_dataset) if label in target_classes]  # type: ignore
target_train_dataset = Subset(train_dataset, target_indices)

print(f"📈 原始训练集大小: {len(train_dataset)}")
print(f"🎯 目标超类训练样本数: {len(target_train_dataset)}")
print(f"📊 占总数据的比例: {len(target_train_dataset)/len(train_dataset)*100:.1f}%")

class TransformedCifar100(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(self.dataset, Subset):
            image, label = self.dataset.dataset[self.dataset.indices[idx]]
        else:
            image, label = self.dataset[idx]
        pixel_values = transform(image).squeeze(0)
        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}

# 创建数据加载器
target_train_ds = TransformedCifar100(target_train_dataset)
full_test_ds = TransformedCifar100(test_dataset)

target_train_loader = DataLoader(target_train_ds, batch_size=BATCH_SIZE, shuffle=True)
full_test_loader = DataLoader(full_test_ds, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. 加载预训练模型并评估基线性能 ---
print("\n🔄 加载预训练模型...")
model = ViTForImageClassification.from_pretrained(
    MODEL_ID,
    num_labels=100,
    ignore_mismatched_sizes=True
).to(device)    

# 评估基线性能
def evaluate_model_detailed(model, test_loader, target_classes):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 计算整体准确率
    overall_accuracy = (all_predictions == all_labels).mean()
    
    # 计算目标超类的性能
    target_mask = np.isin(all_labels, target_classes)
    target_accuracy = (all_predictions[target_mask] == all_labels[target_mask]).mean() if target_mask.sum() > 0 else 0
    
    # 计算非目标超类的性能
    non_target_mask = ~np.isin(all_labels, target_classes)
    non_target_accuracy = (all_predictions[non_target_mask] == all_labels[non_target_mask]).mean() if non_target_mask.sum() > 0 else 0
    
    # 计算目标超类的召回率和精确率
    target_predicted_mask = np.isin(all_predictions, target_classes)
    precision = (np.isin(all_labels[target_predicted_mask], target_classes)).mean() if target_predicted_mask.sum() > 0 else 0
    recall = target_accuracy  # 召回率就是目标超类的准确率
    
    # 计算每个目标类别的详细性能
    target_class_accuracies = {}
    for cls in target_classes:
        cls_mask = all_labels == cls
        if cls_mask.sum() > 0:
            cls_accuracy = (all_predictions[cls_mask] == all_labels[cls_mask]).mean()
            target_class_accuracies[cls] = cls_accuracy
        else:
            target_class_accuracies[cls] = 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'target_accuracy': target_accuracy,
        'non_target_accuracy': non_target_accuracy,
        'target_precision': precision,
        'target_recall': recall,
        'target_samples': target_mask.sum(),
        'non_target_samples': non_target_mask.sum(),
        'target_class_accuracies': target_class_accuracies
    }

print("📊 评估预训练模型的基线性能...")
baseline_results = evaluate_model_detailed(model, full_test_loader, target_classes)

print(f"\n🔍 基线性能 (预训练模型):")
print(f"   整体准确率: {baseline_results['overall_accuracy']:.4f}")
print(f"   目标超类 ({TARGET_SUPERCLASS}) 准确率: {baseline_results['target_accuracy']:.4f}")
print(f"   非目标超类平均准确率: {baseline_results['non_target_accuracy']:.4f}")
print(f"   目标超类精确率: {baseline_results['target_precision']:.4f}")
print(f"   目标超类召回率: {baseline_results['target_recall']:.4f}")
print(f"\n📋 各个类别详细准确率:")
for cls_idx in target_classes:
    cls_name = class_names[cls_idx]
    cls_acc = baseline_results['target_class_accuracies'][cls_idx]
    print(f"   {cls_name}: {cls_acc:.4f}")

# --- 3. 极端偏斜微调 ---
print(f"\n🚀 开始极端偏斜微调 (只使用 {TARGET_SUPERCLASS} 超类数据)...")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

training_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(target_train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(target_train_loader)
    training_losses.append(avg_loss)
    print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

# --- 4. 评估微调后性能 ---
print("\n📊 评估微调后模型性能...")
finetuned_results = evaluate_model_detailed(model, full_test_loader, target_classes)

print(f"\n🎯 微调后性能:")
print(f"   整体准确率: {finetuned_results['overall_accuracy']:.4f}")
print(f"   目标超类 ({TARGET_SUPERCLASS}) 准确率: {finetuned_results['target_accuracy']:.4f}")
print(f"   非目标超类平均准确率: {finetuned_results['non_target_accuracy']:.4f}")
print(f"   目标超类精确率: {finetuned_results['target_precision']:.4f}")
print(f"   目标超类召回率: {finetuned_results['target_recall']:.4f}")
print(f"\n📋 各个类别详细准确率:")
for cls_idx in target_classes:
    cls_name = class_names[cls_idx]
    cls_acc = finetuned_results['target_class_accuracies'][cls_idx]
    print(f"   {cls_name}: {cls_acc:.4f}")

# --- 5. 性能对比分析 ---
print(f"\n📈 性能变化分析:")
overall_change = finetuned_results['overall_accuracy'] - baseline_results['overall_accuracy']
target_change = finetuned_results['target_accuracy'] - baseline_results['target_accuracy']
non_target_change = finetuned_results['non_target_accuracy'] - baseline_results['non_target_accuracy']

print(f"   整体准确率变化: {overall_change:+.4f}")
print(f"   目标超类准确率变化: {target_change:+.4f}")
print(f"   非目标超类准确率变化: {non_target_change:+.4f}")

if target_change > 0:
    print(f"   ✅ 目标超类性能提升了 {target_change:.4f}")
else:
    print(f"   ❌ 目标超类性能下降了 {abs(target_change):.4f}")

if non_target_change > 0:
    print(f"   ✅ 非目标超类性能提升了 {non_target_change:.4f}")
else:
    print(f"   ❌ 非目标超类性能下降了 {abs(non_target_change):.4f}")

# 分析各个类别的变化
print(f"\n📊 目标超类内各类别性能变化:")
for cls_idx in target_classes:
    cls_name = class_names[cls_idx]
    baseline_cls_acc = baseline_results['target_class_accuracies'][cls_idx]
    finetuned_cls_acc = finetuned_results['target_class_accuracies'][cls_idx]
    cls_change = finetuned_cls_acc - baseline_cls_acc
    print(f"   {cls_name}: {baseline_cls_acc:.4f} → {finetuned_cls_acc:.4f} ({cls_change:+.4f})")

# --- 6. 可视化结果 ---
plt.figure(figsize=(15, 10))

# 子图1: 训练损失
plt.subplot(2, 3, 1)
plt.plot(training_losses, 'b-', linewidth=2)
plt.title('训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# 子图2: 准确率对比
plt.subplot(2, 3, 2)
categories = ['整体', '目标超类', '非目标超类']
baseline_values = [baseline_results['overall_accuracy'], 
                  baseline_results['target_accuracy'], 
                  baseline_results['non_target_accuracy']]
finetuned_values = [finetuned_results['overall_accuracy'], 
                   finetuned_results['target_accuracy'], 
                   finetuned_results['non_target_accuracy']]

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, baseline_values, width, label='基线模型', alpha=0.8)
plt.bar(x + width/2, finetuned_values, width, label='微调后模型', alpha=0.8)

plt.xlabel('类别')
plt.ylabel('准确率')
plt.title('性能对比')
plt.xticks(x, categories)
plt.legend()
plt.grid(True, alpha=0.3)

# 子图3: 变化幅度
plt.subplot(2, 3, 3)
changes = [overall_change, target_change, non_target_change]
colors = ['green' if x >= 0 else 'red' for x in changes]
plt.bar(categories, changes, color=colors, alpha=0.7)
plt.xlabel('类别')
plt.ylabel('准确率变化')
plt.title('性能变化 (微调后 - 基线)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)

# 子图4: 目标超类详细分析
plt.subplot(2, 3, 4)
target_metrics = ['准确率', '精确率', '召回率']
baseline_target = [baseline_results['target_accuracy'], 
                  baseline_results['target_precision'], 
                  baseline_results['target_recall']]
finetuned_target = [finetuned_results['target_accuracy'], 
                   finetuned_results['target_precision'], 
                   finetuned_results['target_recall']]

x_metrics = np.arange(len(target_metrics))
plt.bar(x_metrics - width/2, baseline_target, width, label='基线模型', alpha=0.8)
plt.bar(x_metrics + width/2, finetuned_target, width, label='微调后模型', alpha=0.8)

plt.xlabel('指标')
plt.ylabel('分数')
plt.title(f'目标超类 ({TARGET_SUPERCLASS}) 详细分析')
plt.xticks(x_metrics, target_metrics)
plt.legend()
plt.grid(True, alpha=0.3)

# 子图5: 数据分布可视化
plt.subplot(2, 3, 5)
labels = [f'目标超类\n({TARGET_SUPERCLASS})', '其他19个超类']
sizes = [len(target_train_dataset), len(train_dataset) - len(target_train_dataset)]
colors = ['#ff9999', '#66b3ff']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('训练数据分布\n(模拟 beta→0 的极端偏斜)')

# 子图6: 性能总结
plt.subplot(2, 3, 6)
plt.text(0.1, 0.9, f"🎯 实验总结", fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.8, f"目标超类: {TARGET_SUPERCLASS}", fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.7, f"训练样本: {len(target_train_dataset)}", fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.6, f"整体性能变化: {overall_change:+.4f}", fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.5, f"目标超类性能变化: {target_change:+.4f}", fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.4, f"非目标超类性能变化: {non_target_change:+.4f}", fontsize=12, transform=plt.gca().transAxes)

catastrophic_forgetting = abs(non_target_change) > 0.1
if catastrophic_forgetting:
    plt.text(0.1, 0.3, "⚠️  检测到灾难性遗忘", fontsize=12, color='red', transform=plt.gca().transAxes)
else:
    plt.text(0.1, 0.3, "✅ 未发生严重遗忘", fontsize=12, color='green', transform=plt.gca().transAxes)

plt.axis('off')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/extreme_bias_analysis_superclass_{TARGET_SUPERCLASS}.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 7. 保存微调后的模型 ---
model_save_path = f"{SAVE_DIR}/vit_extreme_bias_superclass_{TARGET_SUPERCLASS}"
print(f"\n💾 保存微调后的模型到: {model_save_path}")

model.save_pretrained(model_save_path)
processor.save_pretrained(model_save_path)

# 保存实验结果
results_save_path = f"{SAVE_DIR}/extreme_bias_results_superclass_{TARGET_SUPERCLASS}.txt"
with open(results_save_path, 'w', encoding='utf-8') as f:
    f.write(f"极端数据偏斜实验结果 (beta→0)\n")
    f.write(f"==================================\n\n")
    f.write(f"实验设置:\n")
    f.write(f"  目标超类: {TARGET_SUPERCLASS}\n")
    f.write(f"  包含类别: {', '.join(target_class_names)}\n")
    f.write(f"  训练样本数: {len(target_train_dataset)}\n")
    f.write(f"  训练轮次: {NUM_EPOCHS}\n")
    f.write(f"  学习率: {LEARNING_RATE}\n\n")
    
    f.write(f"基线性能 (预训练模型):\n")
    f.write(f"  整体准确率: {baseline_results['overall_accuracy']:.4f}\n")
    f.write(f"  目标超类准确率: {baseline_results['target_accuracy']:.4f}\n")
    f.write(f"  非目标超类准确率: {baseline_results['non_target_accuracy']:.4f}\n")
    for cls_idx in target_classes:
        cls_name = class_names[cls_idx]
        cls_acc = baseline_results['target_class_accuracies'][cls_idx]
        f.write(f"    {cls_name}: {cls_acc:.4f}\n")
    f.write(f"\n")
    
    f.write(f"微调后性能:\n")
    f.write(f"  整体准确率: {finetuned_results['overall_accuracy']:.4f}\n")
    f.write(f"  目标超类准确率: {finetuned_results['target_accuracy']:.4f}\n")
    f.write(f"  非目标超类准确率: {finetuned_results['non_target_accuracy']:.4f}\n")
    for cls_idx in target_classes:
        cls_name = class_names[cls_idx]
        cls_acc = finetuned_results['target_class_accuracies'][cls_idx]
        f.write(f"    {cls_name}: {cls_acc:.4f}\n")
    f.write(f"\n")
    
    f.write(f"性能变化:\n")
    f.write(f"  整体准确率变化: {overall_change:+.4f}\n")
    f.write(f"  目标超类准确率变化: {target_change:+.4f}\n")
    f.write(f"  非目标超类准确率变化: {non_target_change:+.4f}\n")
    f.write(f"  超类内各类别变化:\n")
    for cls_idx in target_classes:
        cls_name = class_names[cls_idx]
        baseline_cls_acc = baseline_results['target_class_accuracies'][cls_idx]
        finetuned_cls_acc = finetuned_results['target_class_accuracies'][cls_idx]
        cls_change = finetuned_cls_acc - baseline_cls_acc
        f.write(f"    {cls_name}: {cls_change:+.4f}\n")
    f.write(f"\n")
    
    if abs(non_target_change) > 0.1:
        f.write(f"⚠️  检测到严重的灾难性遗忘现象\n")
    else:
        f.write(f"✅ 未发生严重的灾难性遗忘\n")

print(f"📄 实验结果已保存到: {results_save_path}")
print(f"\n🎉 实验完成！您可以在 {SAVE_DIR} 目录中找到:")
print(f"   - 微调后的模型")
print(f"   - 性能分析图表")
print(f"   - 详细结果报告")

print(f"\n🔬 关键发现:")
if target_change > 0.1:
    print(f"   ✅ 目标超类性能显著提升 (+{target_change:.4f})")
if abs(non_target_change) > 0.1:
    print(f"   ⚠️  非目标超类出现明显性能下降 ({non_target_change:+.4f}) - 这是灾难性遗忘的典型表现")
if overall_change < 0:
    print(f"   📉 整体性能下降 ({overall_change:+.4f}) - 说明极端偏斜的负面影响")

print(f"\n💡 这个实验模拟了联邦学习中客户端数据极度不平衡的情况，")
print(f"   有助于理解数据异构性对模型性能的影响。")
