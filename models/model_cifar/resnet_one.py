'''
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
4. NIPS18-Knowledge Distillation by On-the-Fly Native Ensemble
5. NIPS18-Collaborative Learning for Deep Neural Networks

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
class ILR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None
        
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, num_branches = 3, bpscale = False, avg = False, ind = False, zero_init_residual=False, 
        groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.ind = ind
        self.avg = avg
        self.bpscale = bpscale
        self.num_branches = num_branches

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
        
        if self.avg == False:
            self.avgpool_c = nn.AvgPool2d(16)
            self.control_v1 = nn.Linear(fix_inplanes, self.num_branches)
            self.bn_v1 = nn.BatchNorm1d(self.num_branches)
        
        # 自适应加权模块的状态追踪
        self.use_adaptive_weighting = False  # 是否启用自适应加权
        self.epoch_count = 0  # 当前epoch计数（从0开始）
        self.prev_ensem_logits = {}  # 字典：{sample_id: tensor}，存储每个样本上一轮的ensemble logit
        self.current_epoch_ensem_logits = {}  # 字典：存储当前epoch内的ensemble logit

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
        if self.bpscale:
            self.layer_ILR = ILR.apply
                    
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
    
    def compute_wasserstein_dissimilarities(self, logitlist, sample_ids=None):
        """
        计算各分支logit与历史ensemble logit的一阶Wasserstein距离（样本级别）- 优化版本
        """
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
        
        # 优化1: 批量收集历史logits到GPU tensor
        prev_logits_batch = torch.zeros(batch_size, num_classes, device=device)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i, sample_id in enumerate(sample_ids):
            if sample_id in self.prev_ensem_logits:
                prev_logits_batch[i] = self.prev_ensem_logits[sample_id]
                valid_mask[i] = True
        
        # 优化2: 向量化计算所有分支的Wasserstein距离
        current_logits_stacked = torch.stack(logitlist, dim=0)
        current_probs = F.softmax(current_logits_stacked, dim=2)
        prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)
        current_cdfs = torch.cumsum(current_probs, dim=2)
        prev_cdfs = torch.cumsum(prev_probs, dim=2)
        wasserstein_dists = torch.sum(torch.abs(current_cdfs - prev_cdfs), dim=2)
        wasserstein_dists[:, ~valid_mask] = 1.0
        
        # 优化3: 向量化归一化
        dissim_matrix = wasserstein_dists.transpose(0, 1)
        total_weights = dissim_matrix.sum(dim=1, keepdim=True)
        total_weights = torch.where(total_weights > 0, total_weights, torch.ones_like(total_weights))
        normalized_dissim = dissim_matrix / total_weights
        
        return [normalized_dissim[:, i] for i in range(num_branches)]
    
    def update_epoch_history(self):
        """在每个epoch结束时调用，更新历史记录"""
        self.prev_ensem_logits = self.current_epoch_ensem_logits.copy()
        self.current_epoch_ensem_logits = {}
        self.epoch_count += 1
        
    def forward(self, x, sample_ids=None):
        # 生成sample_ids（如果未提供）
        if sample_ids is None:
            batch_size = x.size(0)
            sample_ids = list(range(batch_size))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)            # B x 16 x 32 x 32

        x = self.layer1(x)          # B x 16 x 32 x 32
        x = self.layer2(x)          # B x 32 x 16 x 16
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches) # Backprop rescaling

        x_3 = getattr(self,'layer3_0')(x)   # B x 64 x 8 x 8
        x_3 = self.avgpool(x_3)             # B x 64 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)     # B x 64
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        pro = x_3_1.unsqueeze(-1)        
        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer3_'+str(i))(x)
            temp = self.avgpool(temp)       # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)   
            temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
            temp_1 = temp_1.unsqueeze(-1)
            pro = torch.cat([pro,temp_1],-1)        # B x num_classes x num_branches
        if self.ind:
            return pro, None
        # CL
        else:
            if self.avg:
                x_m = 0
                for i in range(1, self.num_branches):
                    x_m += 1/(self.num_branches-1) * pro[:,:,i]
                x_m = x_m.unsqueeze(-1)
                for i in range(1, self.num_branches):
                    temp = 0
                    for j in range(0, self.num_branches):
                        if j != i:
                            temp += 1/(self.num_branches-1) * pro[:,:,j]       # B x num_classes
                    temp = temp.unsqueeze(-1)
                    x_m = torch.cat([x_m, temp],-1)                            # B x num_classes x num_branches
            # ONE
            else:
                x_c = self.avgpool_c(x)     # B x 32 x 1 x 1
                x_c = x_c.view(x_c.size(0), -1) # B x 32 
                x_c=self.control_v1(x_c)    # B x 3
                x_c=self.bn_v1(x_c)  
                x_c=F.relu(x_c)      
                x_c = F.softmax(x_c, dim=1) # B x 3  
                x_m = x_c[:,0].view(-1, 1).repeat(1, pro[:,:,0].size(1)) * pro[:,:,0]
                for i in range(1, self.num_branches):
                    x_m += x_c[:,i].view(-1, 1).repeat(1, pro[:,:,i].size(1)) * pro[:,:,i]       # B x num_classes
            
            # 自适应加权融合
            if self.use_adaptive_weighting:
                # 将pro转换为logit列表
                logitlist = [pro[:, :, i] for i in range(self.num_branches)]
                
                # 计算相异度权重
                dissimilarities = self.compute_wasserstein_dissimilarities(logitlist, sample_ids)
                
                # 使用权重进行加权融合
                batch_size = pro.size(0)
                num_classes = pro.size(1)
                ensemble_logit = torch.zeros(batch_size, num_classes, device=pro.device)
                
                for i, (logit, weight) in enumerate(zip(logitlist, dissimilarities)):
                    weighted_logit = logit * weight.view(-1, 1)
                    ensemble_logit += weighted_logit
                
                # 存储当前epoch的ensemble输出
                for j, sample_id in enumerate(sample_ids):
                    self.current_epoch_ensem_logits[sample_id] = ensemble_logit[j].detach()
                
                return pro, x_m, ensemble_logit
            
            return pro, x_m

                
def resnet32(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def resnet110(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    
    model = ResNet(Bottleneck, [12, 12, 12], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def wide_resnet20_8(pretrained=False, path=None, **kwargs):
    
    """Constructs a Wide ResNet-28-10 model.
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
