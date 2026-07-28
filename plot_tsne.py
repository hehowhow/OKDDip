"""
在CIFAR-10数据集上绘制t-SNE可视化图
用于展示模型学到的特征表示
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import models.data_loader as data_loader
import models.model_cifar as model_cifar
import utils

# CIFAR-10类别名称
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-100超类名称（如果需要）
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


def extract_features(model, data_loader, device, branch_idx=None, max_samples=None):
    """
    从模型中提取特征
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备（CPU/GPU）
        branch_idx: 要提取的分支索引（0, 1, 2等），如果为None则提取所有分支的平均特征
        max_samples: 最大样本数（用于加速，None表示使用全部数据）
    
    Returns:
        features: numpy数组，shape为[N, feature_dim]
        labels: numpy数组，shape为[N]
    """
    model.eval()
    features_list = []
    labels_list = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if max_samples is not None and sample_count >= max_samples:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播到layer3之前
            x = model.conv1(images)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.layer1(x)
            x = model.layer2(x)
            
            # 提取指定分支的特征
            if branch_idx is not None:
                # 提取特定分支的特征
                x_branch = getattr(model, f'layer3_{branch_idx}')(x)
                x_branch = model.avgpool(x_branch)
                x_branch = x_branch.view(x_branch.size(0), -1)  # [batch_size, 64]
                features = x_branch
            else:
                # 提取所有分支的平均特征
                branch_features = []
                for i in range(model.num_branches):
                    x_branch = getattr(model, f'layer3_{i}')(x)
                    x_branch = model.avgpool(x_branch)
                    x_branch = x_branch.view(x_branch.size(0), -1)
                    branch_features.append(x_branch)
                
                # 平均所有分支的特征
                features = torch.stack(branch_features, dim=0).mean(dim=0)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
            sample_count += images.size(0)
    
    # 合并所有batch
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    
    if max_samples is not None:
        features_array = features_array[:max_samples]
        labels_array = labels_array[:max_samples]
    
    return features_array, labels_array


def plot_tsne(features, labels, class_names, save_path='tsne_visualization.png', 
              title='t-SNE Visualization', perplexity=30, n_iter=1000):
    """
    绘制t-SNE可视化图
    
    Args:
        features: 特征数组，shape为[N, feature_dim]
        labels: 标签数组，shape为[N]
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图表标题
        perplexity: t-SNE的困惑度参数
        n_iter: t-SNE的迭代次数
    """
    print(f"Running t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    
    # 执行t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, 
                random_state=42, verbose=1)
    features_2d = tsne.fit_transform(features)
    
    print("t-SNE completed. Plotting...")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 为每个类别使用不同的颜色和标记
    num_classes = len(class_names)
    
    # 使用tab10和tab20的组合来获得足够的颜色
    if num_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # 为每个类别绘制散点
    for class_idx in range(num_classes):
        mask = labels == class_idx
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=[colors[class_idx]], label=class_names[class_idx],
                  alpha=0.6, s=20, edgecolors='none')
    
    # 设置图表属性
    ax.set_xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # 设置刻度标签
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 设置图例
    if num_classes <= 10:
        ax.legend(loc='best', fontsize=10, framealpha=0.9, 
                 ncol=2, edgecolor='black')
    else:
        # CIFAR-100有太多类别，图例可能会很大
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                 fontsize=8, framealpha=0.9, ncol=2)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"t-SNE visualization saved to {save_path} and PDF version")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot t-SNE visualization for trained model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (e.g., ./CIFAR10/300/GL/resnet32B4T3.0SKLV0/best.pth)')
    parser.add_argument('--model', type=str, default='resnet32',
                       help='Model architecture (default: resnet32)')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       choices=['CIFAR10', 'CIFAR100'],
                       help='Dataset name (default: CIFAR10)')
    parser.add_argument('--num_branches', type=int, default=4,
                       help='Number of branches in the model (default: 4)')
    parser.add_argument('--branch_idx', type=int, default=None,
                       help='Branch index to extract features from (0, 1, 2, etc.). If None, use average of all branches')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for data loading (default: 128)')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples to use (default: 5000, set to -1 for all)')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter (default: 30)')
    parser.add_argument('--n_iter', type=int, default=1000,
                       help='t-SNE number of iterations (default: 1000)')
    parser.add_argument('--gpu_id', type=str, default='0',
                       help='GPU ID (default: 0)')
    parser.add_argument('--data_root', type=str, default='./Data',
                       help='Dataset root directory (default: ./Data)')
    parser.add_argument('--output', type=str, default='tsne_visualization.png',
                       help='Output file path (default: tsne_visualization.png)')
    
    args = parser.parse_args()
    
    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置数据集参数
    if args.dataset == 'CIFAR10':
        num_classes = 10
        class_names = CIFAR10_CLASSES
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        class_names = CIFAR100_CLASSES
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    root = args.data_root
    
    # 加载数据
    print(f"Loading {args.dataset} dataset...")
    _, test_loader = data_loader.dataloader(
        data_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=4,
        root=root
    )
    print("Data loaded successfully.")
    
    # 创建模型
    print(f"Creating {args.model} model with {args.num_branches} branches...")
    model_fd = getattr(model_cifar, 'resnet_GL')
    model = getattr(model_fd, args.model)(
        num_classes=num_classes,
        num_branches=args.num_branches,
        input_channel=utils.lookup(args.model),
        dissimilarity_metric='wasserstein1',
        tau=1.0
    )
    
    # 加载模型权重
    print(f"Loading model weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 处理可能的DataParallel包装
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除'module.'前缀（如果存在）
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
    
    # 提取特征
    max_samples = None if args.max_samples == -1 else args.max_samples
    branch_name = f"Branch {args.branch_idx}" if args.branch_idx is not None else "All Branches (Average)"
    print(f"\nExtracting features from {branch_name}...")
    print(f"Using {'all' if max_samples is None else max_samples} samples from test set")
    
    features, labels = extract_features(
        model, test_loader, device,
        branch_idx=args.branch_idx,
        max_samples=max_samples
    )
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 绘制t-SNE图
    title_branch = f"Branch {args.branch_idx}" if args.branch_idx is not None else "Ensemble"
    title = f"t-SNE Visualization - {args.model} ({title_branch}) on {args.dataset}"
    
    plot_tsne(
        features, labels, class_names,
        save_path=args.output,
        title=title,
        perplexity=args.perplexity,
        n_iter=args.n_iter
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()


