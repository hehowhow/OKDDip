# OKDDip：带跨 Epoch Logit 相异度加权的在线知识蒸馏

本仓库基于 AAAI 2020 的 **Online Knowledge Distillation with Diverse
Peers (OKDDip)**，并加入了基于上一 epoch ensemble logit 相异度的自适应分支加权。

当前实现使用数据集的真实索引跟踪训练样本。即使 `DataLoader` 每个 epoch
重新打乱，同一个索引仍会读取该样本上一 epoch 的 ensemble logit。验证阶段与训练
历史完全隔离，不会再覆盖训练缓存。

## 环境准备

建议使用带 CUDA 的 Python 环境。训练脚本当前直接调用 CUDA，因此不能仅使用 CPU
完成正式训练。

```bash
git clone https://github.com/hehowhow/OKDDip.git
cd OKDDip

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 保存了原项目的复现实验版本。如果这些旧版本无法在当前 Python
上安装，请使用与项目匹配的旧版 Python/CUDA 环境，或安装与本机 CUDA 对应的
PyTorch 和 torchvision，再安装其余依赖。

## 快速启动

CIFAR-10 和 CIFAR-100 会由 torchvision 自动下载到 `--data_root` 指定的目录。

运行 CIFAR-10、ResNet-32：

```bash
python train_GL.py \
  --model resnet32 \
  --dataset CIFAR10 \
  --data_root ./Data \
  --gpu_id 0
```

运行 CIFAR-100，并指定自适应相异度参数：

```bash
python train_GL.py \
  --model resnet32 \
  --dataset CIFAR100 \
  --data_root ./Data \
  --num_branches 4 \
  --dissimilarity_metric wasserstein1 \
  --tau 1.0 \
  --lambda_ensemble 0.5 \
  --gpu_id 0
```

使用多张 GPU：

```bash
python train_GL.py \
  --model resnet32 \
  --dataset CIFAR100 \
  --data_root ./Data \
  --gpu_id 0,1,2,3
```

脚本会根据可见 GPU 数量自动启用 `DataParallel`。样本索引以张量形式随 batch
切分，所有 GPU 的 ensemble logit 会在汇总后统一写入主模型历史缓存。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--model` | `resnet32` | `resnet32`、`resnet110`、`vgg16`、DenseNet 等 |
| `--dataset` | `CIFAR100` | `CIFAR10`、`CIFAR100` 或 `imagenet` |
| `--data_root` | `./Data` | 数据集根目录 |
| `--num_epochs` | `300` | 训练 epoch 数 |
| `--batch_size` | `128` | batch 大小 |
| `--num_branches` | `4` | 分支数量 |
| `--temperature` | `3.0` | 蒸馏温度 |
| `--lambda_ensemble` | `0.5` | ensemble logit 与原教师信号的融合比例 |
| `--dissimilarity_metric` | `wasserstein1` | `wasserstein1`、`wasserstein2`、`euclidean`、`kl` 或 `cosine` |
| `--tau` | `1.0` | 相异度 softmax 的温度 |
| `--schedule` | `150 225` | 学习率衰减 epoch |
| `--gpu_id` | `0` | 可见 GPU，例如 `0` 或 `0,1` |
| `--resume` | 空 | 包含 `last.pth` 的实验目录 |

查看全部参数：

```bash
python train_GL.py --help
```

## 样本历史的工作方式

训练数据返回 `(image, label, dataset_index)`。每个训练 batch 完成前向计算后，
程序以 `dataset_index` 为键保存 ensemble logit：

```text
当前 epoch：样本 index=4217 的各分支 logit
                         │
                         ▼
上一 epoch：样本 index=4217 的 ensemble logit
```

第一轮没有历史记录，分支使用均匀权重。从第二轮开始才使用同一样本上一轮的
ensemble logit 计算相异度。随机裁剪和翻转仍会产生不同增强视图，但底层数据集样本
保持一致。验证前向不读取训练历史，也不写入历史缓存。

## 测试

运行样本索引和历史查找的回归测试：

```bash
python -m unittest tests.test_logit_history
```

进行语法检查：

```bash
python -m compileall models train_GL.py train_GL2.py train_one2.py
```

## 其他训练入口

原始单模型和经典 KD 基线仍可使用：

```bash
python train.py --model resnet32 --dataset CIFAR10
python train_kd.py --model resnet32 --T_model resnet110 \
  --T_model_path ./CIFAR10/resnet110 --dataset CIFAR10
```

ONE 及其自适应变体：

```bash
python train_one.py --model resnet32 --dataset CIFAR10 --data_root ./Data
python train_one2.py --model resnet32 --dataset CIFAR10 --data_root ./Data
```

带自适应历史加权的实验入口包括 `train_GL.py`、`train_GL2.py`、
`train_GL_no_comments.py`、`train_one.py` 和 `train_one2.py`；这些入口均使用稳定
数据集索引。

如需上传实验指标到 Weights & Biases，先安装并登录，然后增加
`--use_wandb`：

```bash
pip install wandb
wandb login
python train_GL.py --dataset CIFAR100 --data_root ./Data --use_wandb
```

凭据由 W&B 登录状态或 `WANDB_API_KEY` 环境变量提供，仓库不会保存 API Key。
更完整的说明见 `WANDB_SETUP.md`。

## 输出目录

训练日志、TensorBoard 事件、checkpoint 和指标默认写入：

```text
<dataset>/<num_epochs>/<type>/<model-and-parameters>/
```

这些文件通常较大，不建议提交到 Git。

## 引用

```bibtex
@inproceedings{chen2020online,
  title={Online Knowledge Distillation with Diverse Peers.},
  author={Chen, Defang and Mei, Jian-Ping and Wang, Can and Feng, Yan and Chen, Chun},
  booktitle={AAAI},
  pages={3430--3437},
  year={2020}
}
```

论文：https://arxiv.org/abs/1912.00350
