'''
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..batch_logit_history import compute_batch_dissimilarities
from ..logit_history import load_history_batch, record_history_batch

__all__ = ['ResNet', 'resnet32', 'resnet110', 'wide_resnet20_8']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_branches = 3, input_channel=64, factor=8, en = False, zero_init_residual=False, 
        groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD = False,
        dissimilarity_metric='wasserstein1', tau=1.0, history_granularity='sample'):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.en = en
        self.num_branches = num_branches
        self.dissimilarity_metric = dissimilarity_metric
        self.tau = tau  # 温度超参数，用于softmax归一化
        if history_granularity not in ('sample', 'batch'):
            raise ValueError(
                "history_granularity must be either 'sample' or 'batch'"
            )
        self.history_granularity = history_granularity
        
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        fix_inplanes=self.inplanes    # 32
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for i in range(num_branches):
            setattr(self, 'layer3_' + str(i), self._make_layer(block, 64, layers[2], stride=2))
            self.inplanes = fix_inplanes  ##reuse self.inplanes
            setattr(self, 'classifier3_' +str(i), nn.Linear(64 * block.expansion, num_classes))
        
        self.query_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        self.key_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        
        # 自适应加权模块的状态追踪
        self.use_adaptive_weighting = True  # 是否启用自适应加权
        self.epoch_count = 0  # 当前epoch计数（从0开始）
        self.prev_ensem_logits = {}  # 字典：{sample_id: tensor}，存储每个样本上一轮的ensemble logit
        self.current_epoch_ensem_logits = {}  # 字典：存储当前epoch内的ensemble logit
        # Batch-level mode keeps O(num_classes + num_branches) state only.
        self.prev_batch_teacher_mean = None
        self.current_epoch_batch_distance_sum = None
        self.current_epoch_batch_distance_count = 0
        self.batch_branch_weights = None
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def compute_dissimilarities(self, logitlist, sample_ids=None):
        """
        计算各分支logit与历史ensemble logit的相异度（样本级别）- 支持多种度量方式
        
        参数:
            logitlist: list of tensors，每个tensor shape为[batch_size, num_classes]
            sample_ids: list，长度为batch_size，每个元素是样本的唯一标识符
        
        返回:
            dissimilarities: list of tensors，长度等于分支数，每个tensor shape为[batch_size]
                            表示该分支每个样本的归一化相异度权重（所有分支权重和为1）
        """
        if self.history_granularity == 'batch':
            return self.compute_batch_level_dissimilarities(logitlist)

        # 第一个epoch：返回均匀权重
        if self.epoch_count == 0 or sample_ids is None or len(self.prev_ensem_logits) == 0:
            batch_size = logitlist[0].size(0)
            num_branches = len(logitlist)
            uniform_weight = 1.0 / num_branches
            return [torch.ones(batch_size, device=logitlist[0].device) * uniform_weight for _ in range(num_branches)]
        
        batch_size = logitlist[0].size(0)
        device = logitlist[0].device
        num_branches = len(logitlist)
        num_classes = logitlist[0].size(1)
        
        # 批量收集历史logits到GPU tensor
        prev_logits_batch, valid_mask = load_history_batch(
            self.prev_ensem_logits,
            sample_ids,
            batch_size,
            num_classes,
            device,
        )
        
        # 堆叠所有分支的logits: [num_branches, batch_size, num_classes]
        current_logits_stacked = torch.stack(logitlist, dim=0)
        
        # 根据不同的度量方式计算相异度
        if self.dissimilarity_metric == 'wasserstein1':
            # 1阶Wasserstein距离（CDF的L1距离）
            current_probs = F.softmax(current_logits_stacked, dim=2)  # [num_branches, batch_size, num_classes]
            prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)  # [1, batch_size, num_classes]
            
            current_cdfs = torch.cumsum(current_probs, dim=2)  # [num_branches, batch_size, num_classes]
            prev_cdfs = torch.cumsum(prev_probs, dim=2)  # [1, batch_size, num_classes]
            
            dissim_dists = torch.sum(torch.abs(current_cdfs - prev_cdfs), dim=2)  # [num_branches, batch_size]
            
        elif self.dissimilarity_metric == 'wasserstein2':
            # 2阶Wasserstein距离（CDF的L2距离）
            current_probs = F.softmax(current_logits_stacked, dim=2)  # [num_branches, batch_size, num_classes]
            prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)  # [1, batch_size, num_classes]
            
            current_cdfs = torch.cumsum(current_probs, dim=2)  # [num_branches, batch_size, num_classes]
            prev_cdfs = torch.cumsum(prev_probs, dim=2)  # [1, batch_size, num_classes]
            
            dissim_dists = torch.sqrt(torch.sum((current_cdfs - prev_cdfs)**2, dim=2))  # [num_branches, batch_size]
            
        elif self.dissimilarity_metric == 'euclidean':
            # 欧氏距离（在概率分布上）
            current_probs = F.softmax(current_logits_stacked, dim=2)  # [num_branches, batch_size, num_classes]
            prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)  # [1, batch_size, num_classes]
            
            dissim_dists = torch.sqrt(torch.sum((current_probs - prev_probs)**2, dim=2))  # [num_branches, batch_size]
            
        elif self.dissimilarity_metric == 'kl':
            # KL散度 KL(current || prev)
            current_probs = F.softmax(current_logits_stacked, dim=2)  # [num_branches, batch_size, num_classes]
            prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)  # [1, batch_size, num_classes]
            
            # 添加小的epsilon避免log(0)
            eps = 1e-8
            current_probs = current_probs + eps
            prev_probs = prev_probs + eps
            
            # KL(current || prev) = sum(current * log(current / prev))
            dissim_dists = torch.sum(current_probs * torch.log(current_probs / prev_probs), dim=2)  # [num_branches, batch_size]
            
        elif self.dissimilarity_metric == 'cosine':
            # 余弦相似度转相异度（1 - cosine_similarity）
            current_probs = F.softmax(current_logits_stacked, dim=2)  # [num_branches, batch_size, num_classes]
            prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)  # [1, batch_size, num_classes]
            
            # 计算余弦相似度
            numerator = torch.sum(current_probs * prev_probs, dim=2)  # [num_branches, batch_size]
            current_norm = torch.sqrt(torch.sum(current_probs**2, dim=2))  # [num_branches, batch_size]
            prev_norm = torch.sqrt(torch.sum(prev_probs**2, dim=2))  # [1, batch_size]
            
            cosine_sim = numerator / (current_norm * prev_norm + 1e-8)
            dissim_dists = 1 - cosine_sim  # 转换为相异度
            
        else:
            raise ValueError(f"Unknown dissimilarity metric: {self.dissimilarity_metric}")
        
        # 处理无历史记录的样本（设为1.0）- 使用非就地操作避免梯度计算错误
        # 将 valid_mask 扩展到 [num_branches, batch_size] 的形状
        valid_mask_expanded = valid_mask.unsqueeze(0).expand(dissim_dists.shape[0], -1)
        dissim_dists = torch.where(valid_mask_expanded, dissim_dists, torch.ones_like(dissim_dists))
        
        # 转置以便按样本归一化: [batch_size, num_branches]
        dissim_matrix = dissim_dists.transpose(0, 1)
        
        # 使用softmax归一化（带温度参数τ）
        # w_m^i = exp(d_m^i / τ) / Σ_j exp(d_j^i / τ)
        # F.softmax会自动处理数值稳定性（减去最大值）
        normalized_dissim = F.softmax(dissim_matrix / self.tau, dim=1)  # [batch_size, num_branches]
        
        # 转换为list格式（与原接口兼容）
        return [normalized_dissim[:, i] for i in range(num_branches)]

    def compute_batch_level_dissimilarities(self, logitlist):
        """Return one fixed branch-weight vector for the whole current epoch.

        Epochs 0 and 1 use uniform weights. During epoch 1 and later, each
        training batch compares every branch's mean student logit with the
        immediately preceding training batch's mean teacher logit. Distances
        are averaged over the epoch and converted into the weights used by the
        next epoch.
        """
        batch_size = logitlist[0].size(0)
        device = logitlist[0].device
        dtype = logitlist[0].dtype
        num_branches = len(logitlist)

        if self.epoch_count < 2 or self.batch_branch_weights is None:
            weights = torch.full(
                (num_branches,),
                1.0 / num_branches,
                device=device,
                dtype=dtype,
            )
        else:
            weights = self.batch_branch_weights.to(device=device, dtype=dtype)
            if weights.numel() != num_branches:
                raise RuntimeError(
                    "Stored batch branch weights do not match model branches"
                )

        return [
            weights[i].expand(batch_size)
            for i in range(num_branches)
        ]
    
    def update_epoch_history(self):
        """
        在每个epoch结束时调用，更新历史记录
        """
        if self.history_granularity == 'batch':
            if (self.epoch_count >= 1
                    and self.current_epoch_batch_distance_count > 0):
                mean_distances = (
                    self.current_epoch_batch_distance_sum
                    / self.current_epoch_batch_distance_count
                )
                self.batch_branch_weights = F.softmax(
                    mean_distances / self.tau, dim=0
                ).detach()
            self.current_epoch_batch_distance_sum = None
            self.current_epoch_batch_distance_count = 0
            self.epoch_count += 1
            return

        # 将当前epoch的输出作为下一轮的历史参考
        self.prev_ensem_logits = self.current_epoch_ensem_logits.copy()
        # 清空当前记录
        self.current_epoch_ensem_logits = {}
        # 更新epoch计数
        self.epoch_count += 1

    def record_epoch_logits(
            self, sample_ids, ensemble_logits, branch_logits=None):
        """Record gathered training logits under stable dataset indices."""
        if self.history_granularity == 'batch':
            if branch_logits is None:
                raise ValueError(
                    "batch history requires gathered branch_logits"
                )
            if (self.epoch_count >= 1
                    and self.prev_batch_teacher_mean is not None):
                with torch.no_grad():
                    branch_mean_logits = (
                        branch_logits.detach().mean(dim=0).transpose(0, 1)
                    )
                    teacher_mean = self.prev_batch_teacher_mean.to(
                        device=branch_logits.device,
                        dtype=branch_logits.dtype,
                    )
                    distances = compute_batch_dissimilarities(
                        branch_mean_logits,
                        teacher_mean,
                        self.dissimilarity_metric,
                    )
                    if self.current_epoch_batch_distance_sum is None:
                        self.current_epoch_batch_distance_sum = distances
                    else:
                        self.current_epoch_batch_distance_sum = (
                            self.current_epoch_batch_distance_sum.to(
                                branch_logits.device
                            )
                            + distances
                        )
                    self.current_epoch_batch_distance_count += 1
            self.prev_batch_teacher_mean = (
                ensemble_logits.detach().mean(dim=0)
            )
            return
        record_history_batch(
            self.current_epoch_ensem_logits, sample_ids, ensemble_logits
        )

    def get_batch_branch_weights(self):
        """Return current batch-level weights as a CPU list for logging."""
        if self.history_granularity != 'batch':
            return None
        num_branches = self.num_branches if self.en else self.num_branches - 1
        if self.epoch_count < 2 or self.batch_branch_weights is None:
            return [1.0 / num_branches] * num_branches
        return self.batch_branch_weights.detach().cpu().tolist()
        
    def forward(self, x, sample_ids=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)            # B x 16 x 32 x 32

        x = self.layer1(x)          # B x 16 x 32 x 32
        x = self.layer2(x)          # B x 32 x 16 x 16
        x_3 = getattr(self,'layer3_0')(x)   # B x 64 x 8 x 8
        x_3 = self.avgpool(x_3)             # B x 64 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)     # B x 64
        proj_q = self.query_weight(x_3)     # B x 8
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_3)       # B x 8  
        proj_k = proj_k[:, None, :]
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        pro = x_3_1.unsqueeze(-1) 
        if self.en:
            for i in range(1, self.num_branches):
                temp = getattr(self, 'layer3_'+str(i))(x)
                temp = self.avgpool(temp)       # B x 64 x 1 x 1
                temp = temp.view(temp.size(0), -1)   
                temp_q = self.query_weight(temp)
                temp_k = self.key_weight(temp)
                temp_q = temp_q[:, None, :]
                temp_k = temp_k[:, None, :]
                temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
                temp_1 = temp_1.unsqueeze(-1)
                pro = torch.cat([pro,temp_1],-1)        # B x num_classes x num_branches
                proj_q = torch.cat([proj_q, temp_q], 1) # B x num_branches x 8
                proj_k = torch.cat([proj_k, temp_k], 1) 
            
            # 原始注意力机制
            energy = torch.bmm(proj_q, proj_k.permute(0,2,1)) 
            attention = F.softmax(energy, dim = -1) 
            x_m = torch.bmm(pro, attention.permute(0,2,1))
            
            # 自适应加权融合
            if self.use_adaptive_weighting:
                # 将pro转换为logit列表：[batch_size, num_classes, num_branches] -> list of [batch_size, num_classes]
                logitlist = [pro[:, :, i] for i in range(self.num_branches)]
                
                # 计算相异度权重
                dissimilarities = self.compute_dissimilarities(logitlist, sample_ids)
                
                # 使用权重进行加权融合
                batch_size = pro.size(0)
                num_classes = pro.size(1)
                ensemble_logit = torch.zeros(batch_size, num_classes, device=pro.device)
                
                for i, (logit, weight) in enumerate(zip(logitlist, dissimilarities)):
                    weighted_logit = logit * weight.view(-1, 1)  # [batch_size, num_classes]
                    ensemble_logit += weighted_logit
                
                # 返回：pro包含各分支logits，ensemble_logit是自适应加权后的结果
                return pro, x_m, ensemble_logit
            
            return pro, x_m
        else:
            for i in range(1, self.num_branches - 1):
                temp = getattr(self, 'layer3_'+str(i))(x)
                temp = self.avgpool(temp)       # B x 64 x 1 x 1
                temp = temp.view(temp.size(0), -1)   
                temp_q = self.query_weight(temp)
                temp_k = self.key_weight(temp)
                temp_q = temp_q[:, None, :]
                temp_k = temp_k[:, None, :]
                temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
                temp_1 = temp_1.unsqueeze(-1)
                pro = torch.cat([pro,temp_1],-1)        # B x num_classes x num_branches
                proj_q = torch.cat([proj_q, temp_q], 1) # B x num_branches x 8
                proj_k = torch.cat([proj_k, temp_k], 1) 
            
            # 原始注意力机制
            energy =  torch.bmm(proj_q, proj_k.permute(0,2,1)) 
            attention = F.softmax(energy, dim = -1) 
            x_m = torch.bmm(pro, attention.permute(0,2,1))
            
            temp = getattr(self, 'layer3_'+str(self.num_branches - 1))(x)
            temp = self.avgpool(temp)       # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)   
            temp_out = getattr(self, 'classifier3_' + str(self.num_branches - 1))(temp)
            
            # 自适应加权融合（仅对前num_branches-1个分支）
            if self.use_adaptive_weighting:
                # 将pro转换为logit列表（不包括最后一个分支）
                logitlist = [pro[:, :, i] for i in range(self.num_branches - 1)]
                
                # 计算相异度权重
                dissimilarities = self.compute_dissimilarities(logitlist, sample_ids)
                
                # 使用权重进行加权融合
                batch_size = pro.size(0)
                num_classes = pro.size(1)
                ensemble_logit = torch.zeros(batch_size, num_classes, device=pro.device)
                
                for i, (logit, weight) in enumerate(zip(logitlist, dissimilarities)):
                    weighted_logit = logit * weight.view(-1, 1)  # [batch_size, num_classes]
                    ensemble_logit += weighted_logit
                
                return pro, x_m, temp_out, ensemble_logit
            
            return pro, x_m, temp_out
        
def resnet32(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def resnet110(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = ResNet(Bottleneck, [12, 12, 12], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def wide_resnet20_8(pretrained=False, progress=True, **kwargs):
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet(Bottleneck, [2, 2, 2], width_per_group = 64 * 8, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
