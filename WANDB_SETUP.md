# Weights & Biases (W&B) 使用指南

本文档介绍如何在远程服务器上使用 Weights & Biases 来监控训练过程。

## 1. 安装 wandb

在远程服务器上安装 wandb：

```bash
pip install wandb
```

## 2. 登录 W&B 账户

### 方法一：在远程服务器上直接登录（推荐）

```bash
wandb login
```

系统会提示你输入 API key。你可以从以下地址获取：
https://wandb.ai/authorize

复制 API key 并粘贴到终端（注意：粘贴时不会显示字符）。

### 方法二：使用环境变量

如果无法交互式登录，可以设置环境变量：

```bash
export WANDB_API_KEY=你的API_KEY
```

或者在训练脚本启动前添加：

```bash
WANDB_API_KEY=你的API_KEY python train_GL.py --use_wandb ...
```

### 方法三：离线模式（无需登录）

如果服务器无法访问互联网，可以使用离线模式：

```bash
export WANDB_MODE=offline
```

离线模式会将日志保存到本地，之后可以同步到 W&B：

```bash
wandb sync wandb/run-xxx
```

## 3. 运行训练并启用 W&B

在原有的训练命令后添加 `--use_wandb` 参数：

```bash
# 基础用法
python train_GL.py --use_wandb --dataset CIFAR100 --model resnet32

# 指定项目名称
python train_GL.py --use_wandb --wandb_project MyProject --dataset CIFAR100

# 指定团队/用户名
python train_GL.py --use_wandb --wandb_entity your-username --dataset CIFAR100

# 完整示例
python train_GL.py \
    --use_wandb \
    --wandb_project OKDDip-GL \
    --wandb_entity your-username \
    --dataset CIFAR100 \
    --model resnet32 \
    --num_branches 4 \
    --lambda_ensemble 0.5 \
    --gpu_id 0
```

## 4. 在网页上查看训练情况

### 实时查看

1. 打开浏览器访问：https://wandb.ai
2. 登录你的账户
3. 在左侧菜单选择 "Projects"
4. 找到你的项目（默认为 `OKDDip-GL`）
5. 点击进入项目，即可看到正在运行的实验

### 查看的内容包括：

- **实时指标图表**：
  - 训练/测试损失曲线
  - 各分支的准确率
  - 学习率变化
  - Consistency weight
  - 分支多样性 (diversity)

- **系统监控**：
  - GPU 使用率
  - 内存使用
  - CPU 使用率

- **配置信息**：
  - 所有超参数
  - 模型结构
  - 数据集信息

- **日志输出**：
  - 训练过程的输出日志

## 5. W&B 核心功能

### 5.1 比较多个实验

在项目页面可以：
- 选择多个 runs 进行对比
- 并排查看不同配置的结果
- 创建自定义对比图表

### 5.2 超参数重要性分析

W&B 可以自动分析哪些超参数对结果影响最大：
- 点击 "Sweeps" 标签
- 查看参数重要性排名

### 5.3 生成报告

- 点击 "Reports" 创建可分享的报告
- 添加图表、表格和说明
- 分享给团队成员或公开

### 5.4 设置告警

- 在 run 页面点击 "Alerts"
- 设置条件（如准确率达到某个值）
- 可以通过邮件或 Slack 接收通知

## 6. 常见问题解决

### 问题1：连接超时

如果服务器网络不稳定：

```bash
# 增加超时时间
export WANDB_CONSOLE_TIMEOUT=60

# 或使用离线模式
export WANDB_MODE=offline
```

### 问题2：磁盘空间不足

W&B 会在本地缓存数据，可以限制缓存大小：

```bash
# 限制缓存为 1GB
export WANDB_CACHE_DIR=/path/to/cache
export WANDB_CACHE_SIZE=1073741824
```

### 问题3：已有 API key 但仍提示登录

```bash
# 重新登录
wandb login --relogin

# 或者清除旧的配置
rm ~/.netrc
wandb login
```

### 问题4：无法访问网页

确保：
1. 你的浏览器可以访问 https://wandb.ai
2. 账户已经登录
3. 训练脚本已经启动并开始记录日志
4. 刷新网页查看最新数据

## 7. SSH 端口转发（可选）

如果你的远程服务器有防火墙限制，可以使用 SSH 端口转发：

```bash
# 在本地电脑上运行
ssh -L 8080:localhost:8080 user@remote-server

# 然后在浏览器访问
http://localhost:8080
```

但对于 W&B，通常不需要端口转发，因为数据是上传到 wandb.ai 云端的。

## 8. 高级配置

### 自定义记录频率

如果觉得日志太频繁或太稀疏，可以调整：

```python
# 在代码中设置
wandb.init(
    project="OKDDip-GL",
    config=vars(args),
    settings=wandb.Settings(
        _stats_sample_rate_seconds=30,  # 系统监控采样间隔
        _stats_samples_to_average=10,   # 平均采样数
    )
)
```

### 保存模型文件

W&B 可以自动保存最佳模型：

```python
# 在保存最佳模型时
if is_best:
    wandb.save("best.pth")
    wandb.log({"best_acc": best_acc})
```

### 记录示例图片

可以记录一些示例预测结果：

```python
images = wandb.Image(img_tensor, caption="Sample predictions")
wandb.log({"examples": images})
```

## 9. 移动端查看

W&B 也有移动 App：
- iOS: 在 App Store 搜索 "Weights & Biases"
- Android: 在 Google Play 搜索 "Weights & Biases"

可以随时随地查看训练进度！

## 10. 命令行快速参考

```bash
# 登录
wandb login

# 查看当前状态
wandb status

# 同步离线运行
wandb sync wandb/run-xxx

# 查看本地运行历史
wandb sync --show

# 清理本地缓存
wandb artifact cache cleanup 1GB
```

## 联系支持

- W&B 文档：https://docs.wandb.ai/
- 社区论坛：https://community.wandb.ai/
- GitHub Issues：https://github.com/wandb/wandb/issues

---

祝训练顺利！🚀

