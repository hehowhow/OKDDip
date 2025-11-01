'''
VGG16 for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['vgg16', 'vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}

class VGG(nn.Module):
    def __init__(self, num_classes=10, num_branches=3, factor = 8, en= False, depth=16, dropout = 0.5):
        super(VGG, self).__init__()
        self.inplances = 64
        self.en = en
        self.num_branches = num_branches
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(inplace=True)            
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
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
            nn.Dropout(p = dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, num_classes),
            ))
    
        input_channel = 512
        self.query_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        self.key_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        
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
        prev_logits_batch = torch.zeros(batch_size, num_classes, device=device)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i, sample_id in enumerate(sample_ids):
            if sample_id in self.prev_ensem_logits:
                prev_logits_batch[i] = self.prev_ensem_logits[sample_id]
                valid_mask[i] = True
        
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
    
    def forward(self, x, sample_ids=None):
        # 生成sample_ids（如果未提供）
        if sample_ids is None:
            batch_size = x.size(0)
            # 使用batch内的索引作为临时ID（训练时会被覆盖）
            sample_ids = list(range(batch_size))
    
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
        x_3 = getattr(self,'layer3_0')(x)   # B x 512 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)     # B x 512
        proj_q = self.query_weight(x_3)     # B x 64
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_3)       # B x 64 
        proj_k = proj_k[:, None, :]
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        pro = x_3_1.unsqueeze(-1)        
        if self.en:
            for i in range(1, self.num_branches):
                temp = getattr(self, 'layer3_'+str(i))(x)
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
                dissimilarities = self.compute_wasserstein_dissimilarities(logitlist, sample_ids)
                
                # 使用权重进行加权融合
                batch_size = pro.size(0)
                num_classes = pro.size(1)
                ensemble_logit = torch.zeros(batch_size, num_classes, device=pro.device)
                
                for i, (logit, weight) in enumerate(zip(logitlist, dissimilarities)):
                    weighted_logit = logit * weight.view(-1, 1)  # [batch_size, num_classes]
                    ensemble_logit += weighted_logit
                
                # 存储当前epoch的ensemble输出
                for j, sample_id in enumerate(sample_ids):
                    self.current_epoch_ensem_logits[sample_id] = ensemble_logit[j].detach()
                
                # 返回：pro包含各分支logits，ensemble_logit是自适应加权后的结果
                return pro, x_m, ensemble_logit
            
            return pro, x_m
        else:
            for i in range(1, self.num_branches - 1):
                temp = getattr(self, 'layer3_'+str(i))(x)
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
                
                # 存储当前epoch的ensemble输出
                for j, sample_id in enumerate(sample_ids):
                    self.current_epoch_ensem_logits[sample_id] = ensemble_logit[j].detach()
                
                return pro, x_m, temp_out, ensemble_logit
            
            return pro, x_m, temp_out

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
