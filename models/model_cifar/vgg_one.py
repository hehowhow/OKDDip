'''
VGG16 for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
2. NIPS18-Knowledge Distillation by On-the-Fly Native Ensemble
3. NIPS18-Collaborative Learning for Deep Neural Networks

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['vgg16', 'vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}

class ILR(torch.autograd.Function):
   
    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output/num_branches, None


class VGG(nn.Module):
    def __init__(self, num_classes=10, num_branches=3, bpscale = False, avg = False, ind = False, depth=16):
        super(VGG, self).__init__()
        self.inplances = 64
        self.avg = avg
        self.bpscale = bpscale
        self.num_branches = num_branches
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ind = ind
        
        self.layer1 = self._make_layers(128, 2)
        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4
        
        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        for i in range(num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layers(512, num_layer))
            setattr(self, 'classifier3_'+str(i), nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),))
            
        if self.avg == False:
            self.avgpool_c = nn.AdaptiveAvgPool2d((1,1))
            self.control_v1 = nn.Linear(self.inplances, self.num_branches)
            self.bn_v1 = nn.BatchNorm1d(self.num_branches)
        if self.bpscale:
            self.layer_ILR = ILR.apply
        
        # 自适应加权模块的状态追踪
        self.use_adaptive_weighting = True  # 是否启用自适应加权
        self.epoch_count = 0  # 当前epoch计数（从0开始）
        self.prev_ensem_logits = {}  # 字典：{sample_id: tensor}，存储每个样本上一轮的ensemble logit
        self.current_epoch_ensem_logits = {}  # 字典：存储当前epoch内的ensemble logit
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _make_layers(self, input, num_layer):    
        layers=[]
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.bpscale:
            x = self.layer_ILR(x, self.num_branches) # Backprop rescaling
            
        x_3 = getattr(self,'layer3_0')(x)   # B x 64 x 8 x 8
        x_3 = x_3.view(x_3.size(0), -1)     # B x 64
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        pro = x_3_1.unsqueeze(-1)        
        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer3_'+str(i))(x)
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
                x_c=self.avgpool_c(x)
                x_c = x_c.view(x_c.size(0), -1) # B x 32 
                x_c=self.control_v1(x_c)    # B x 3
                x_c=self.bn_v1(x_c)  
                x_c=F.relu(x_c)      
                x_c = F.softmax(x_c, dim=1) # B x 3  
            
                x_3 = getattr(self,'layer3_0')(x)   # B x 64 x 8 x 8
                x_3 = x_3.view(x_3.size(0), -1)     # B x 64
                x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
                x_m = x_c[:,0].view(-1, 1).repeat(1, x_3_1.size(1)) * x_3_1
                pro = x_3_1.unsqueeze(-1) 
                for i in range(1, self.num_branches):
                    temp = getattr(self, 'layer3_'+str(i))(x)
                    temp = temp.view(temp.size(0), -1)   
                    temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
                    x_m += x_c[:,i].view(-1, 1).repeat(1, temp_1.size(1)) * temp_1       # B x num_classes
                    temp_1 = temp_1.unsqueeze(-1)
                    pro = torch.cat([pro,temp_1],-1)        # B x num_classes x num_branches
            
            # 自适应加权逻辑
            if self.use_adaptive_weighting:
                # 提取各分支的logits
                logitlist = [pro[:, :, i] for i in range(self.num_branches)]
                
                # 计算自适应权重（基于与历史ensemble的相异度）
                adaptive_weights = self.compute_wasserstein_dissimilarities(logitlist, sample_ids)
                
                # 计算加权ensemble logit
                ensemble_logit = sum(w.unsqueeze(1) * logit for w, logit in zip(adaptive_weights, logitlist))
                
                # 存储当前epoch的ensemble logit（用于下一个epoch）
                if sample_ids is not None:
                    for i, sample_id in enumerate(sample_ids):
                        self.current_epoch_ensem_logits[sample_id] = ensemble_logit[i].detach().clone()
                
                return pro, x_m, ensemble_logit
            else:
                return pro, x_m
    
def vgg16(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG16 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
    
def vgg19(pretrained=False, path=None, **kwargs):
    """
    Constructs a VGG19 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = VGG(depth=19, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
