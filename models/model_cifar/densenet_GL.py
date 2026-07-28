'''
DenseNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
2. https://github.com/liuzhuang13/DenseNet
3. https://github.com/gpleiss/efficient_densenet_pytorch
4. Gao Huang, zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
Densely Connetcted Convolutional Networks. https://arxiv.org/abs/1608.06993

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from ..logit_history import load_history_batch, record_history_batch

__all__ = ['DenseNet', 'densenetd40k12', 'densenetd100k12']

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

class ILR(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, num_branches):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        num_branches = ctx.num_branches
        return grad_output/num_branches, None

        
class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), num_branches = 3, bpscale = False, input_channel= 132, factor = 8, compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7
        self.num_branches = num_branches
        self.bpscale = bpscale
        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i != len(block_config) - 1:
                block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
            else:                
                block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
                )
                for i in range(self.num_branches):
                    setattr(self, 'Branch' + str(i), block)
        # Final batch norm
        #self.features.add_module('norm_final', nn.BatchNorm2d(num_features))    # optional
        #self.features.add_module('relu_final', nn.ReLU(inplace = True))         # optional        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        num_features = num_features + num_layers * growth_rate
        for i in range(self.num_branches):
            setattr(self, 'norm_final_' + str(i), nn.BatchNorm2d(num_features))
            setattr(self, 'relu_final_' + str(i), nn.ReLU(inplace = True))
        # Linear layer
        for i in range(self.num_branches):
            setattr(self, 'classifier3_' + str(i), nn.Linear(num_features, num_classes))
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
        self.query_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        self.key_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        
        # 自适应加权模块的状态追踪
        self.use_adaptive_weighting = True  # 是否启用自适应加权
        self.epoch_count = 0  # 当前epoch计数（从0开始）
        self.prev_ensem_logits = {}  # 字典：{sample_id: tensor}，存储每个样本上一轮的ensemble logit
        self.current_epoch_ensem_logits = {}  # 字典：存储当前epoch内的ensemble logit
        
        if self.bpscale:
            self.layer_ILR = ILR.apply
    
    def compute_wasserstein_dissimilarities(self, logitlist, sample_ids=None):
        """
        计算各分支logit与历史ensemble logit的一阶Wasserstein距离（样本级别）- 优化版本
        
        参数:
            logitlist: list of tensors，每个tensor shape为[batch_size, num_classes]
            sample_ids: list，长度为batch_size，每个元素是样本的唯一标识符
        
        返回:
            dissimilarities: list of tensors，长度等于分支数，每个tensor shape为[batch_size]
                            表示该分支每个样本的归一化相异度权重（所有分支权重和为1）
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
        prev_logits_batch, valid_mask = load_history_batch(
            self.prev_ensem_logits,
            sample_ids,
            batch_size,
            num_classes,
            device,
        )
        
        # 优化2: 向量化计算所有分支的Wasserstein距离
        # 堆叠所有分支的logits: [num_branches, batch_size, num_classes]
        current_logits_stacked = torch.stack(logitlist, dim=0)
        
        # 批量计算概率分布和CDF
        current_probs = F.softmax(current_logits_stacked, dim=2)  # [num_branches, batch_size, num_classes]
        prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)  # [1, batch_size, num_classes]
        
        current_cdfs = torch.cumsum(current_probs, dim=2)  # [num_branches, batch_size, num_classes]
        prev_cdfs = torch.cumsum(prev_probs, dim=2)  # [1, batch_size, num_classes]
        
        # 批量计算Wasserstein距离: [num_branches, batch_size]
        wasserstein_dists = torch.sum(torch.abs(current_cdfs - prev_cdfs), dim=2)
        
        # 处理无历史记录的样本（设为1.0）
        wasserstein_dists[:, ~valid_mask] = 1.0
        
        # 优化3: 向量化归一化
        # 转置以便按样本归一化: [batch_size, num_branches]
        dissim_matrix = wasserstein_dists.transpose(0, 1)
        
        # 批量归一化
        total_weights = dissim_matrix.sum(dim=1, keepdim=True)  # [batch_size, 1]
        total_weights = torch.where(total_weights > 0, total_weights, torch.ones_like(total_weights))
        normalized_dissim = dissim_matrix / total_weights  # [batch_size, num_branches]
        
        # 转换为list格式（与原接口兼容）
        return [normalized_dissim[:, i] for i in range(num_branches)]
    
    def update_epoch_history(self):
        """
        在每个epoch结束时调用，更新历史记录
        """
        # 将当前epoch的输出作为下一轮的历史参考
        self.prev_ensem_logits = self.current_epoch_ensem_logits.copy()
        # 清空当前记录
        self.current_epoch_ensem_logits = {}
        # 更新epoch计数
        self.epoch_count += 1
    def record_epoch_logits(self, sample_ids, ensemble_logits):
        """Record gathered training logits under stable dataset indices."""
        record_history_batch(
            self.current_epoch_ensem_logits, sample_ids, ensemble_logits
        )
            
    def forward(self, x, sample_ids=None):
        # For depth 40 growth_rate 1      B x 3 x 32 x 32
        x = self.features(x)            # B x 60 x 8 x 8 
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches)
            
        x_3 = getattr(self, 'Branch0')(x)         # B x 132 x 8 x 8
        x_3 = getattr(self, 'norm_final_0')(x_3)
        x_3 = getattr(self, 'relu_final_0')(x_3)
        x_3 = self.avgpool(x_3).view(x_3.size(0), -1)         # B x 132 
        proj_q = self.query_weight(x_3)     # B x 8
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_3)       # B x 8  
        proj_k = proj_k[:, None, :]
        x_3_1 = getattr(self, 'classifier3_0')(x_3)                         # B x num_classes
        pro = x_3_1.unsqueeze(-1)
        for i in range(1, self.num_branches):
            temp = getattr(self, 'Branch' + str(i))(x)
            temp = getattr(self, 'norm_final_' + str(i))(temp)
            temp = getattr(self, 'relu_final_' + str(i))(temp)
            temp = self.avgpool(temp).view(temp.size(0), -1)         # B x 132 
            temp_q = self.query_weight(temp)
            temp_k = self.key_weight(temp)
            temp_q = temp_q[:, None, :]
            temp_k = temp_k[:, None, :]
            temp_1 = getattr(self, 'classifier3_' + str(i))(temp)      # B x num_classes
            temp_1 = temp_1.unsqueeze(-1)
            pro = torch.cat([pro, temp_1], -1)
            proj_q = torch.cat([proj_q, temp_q], 1) # B x num_branches x 8
            proj_k = torch.cat([proj_k, temp_k], 1) 
            
        # 原始注意力机制
        energy =  torch.bmm(proj_q, proj_k.permute(0,2,1)) 
        attention = F.softmax(energy, dim = -1) 
        x_m = torch.bmm(pro, attention.permute(0,2,1))
        
        temp = getattr(self, 'Branch'+str(self.num_branches - 1))(x)
        temp = self.avgpool(temp)       # B x 64 x 1 x 1
        temp = temp.view(temp.size(0), -1)   
        temp_out = getattr(self, 'classifier3_' + str(self.num_branches - 1))(temp)
        
        # 自适应加权融合（仅对前num_branches-1个分支）
        if self.use_adaptive_weighting:
            # 将pro转换为logit列表（不包括最后一个分支）
            logitlist = [pro[:, :, i] for i in range(self.num_branches - 1)]
            
            # 计算相异度权重
            dissimilarities = self.compute_wasserstein_dissimilarities(logitlist, sample_ids)
            
            # 使用权重进行加权融合
            batch_size = pro.size(0)
            num_classes = pro.size(1)
            ensemble_logit = torch.zeros(batch_size, num_classes, device=pro.device)
            
            for i, (logit, weight) in enumerate(zip(logitlist, dissimilarities)):
                weighted_logit = logit * weight.view(-1, 1)  # [batch_size, num_classes]
                ensemble_logit += weighted_logit
            
            return pro, x_m, temp_out, ensemble_logit
        
        return pro, x_m, temp_out

        # features = self.features(x)
        # out = F.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        # out = self.classifier(out)
        # return out
        
def densenetd40k12(pretrained=False, path=None, **kwargs):
    """
    Constructs a densenetD40K12 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    
    model = DenseNet(growth_rate = 12, block_config = [6,6,6], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def densenetd100k12(pretrained=False, path=None, **kwargs):
    """
    Constructs a densenetD100K12 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    
    model = DenseNet(growth_rate = 12, block_config = [16,16,16], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
