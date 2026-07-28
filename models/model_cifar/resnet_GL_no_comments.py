import torch
import torch.nn as nn
import torch.nn.functional as F
from ..logit_history import load_history_batch, record_history_batch

__all__ = ['ResNet', 'resnet32', 'resnet110', 'wide_resnet20_8']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
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
        groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD = False, dissimilarity_metric='wasserstein1', tau=1.0):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.en = en
        self.num_branches = num_branches
        self.dissimilarity_metric = dissimilarity_metric
        self.tau = tau
        
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
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
        fix_inplanes=self.inplanes
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for i in range(num_branches):
            setattr(self, 'layer3_' + str(i), self._make_layer(block, 64, layers[2], stride=2))
            self.inplanes = fix_inplanes
            setattr(self, 'classifier3_' +str(i), nn.Linear(64 * block.expansion, num_classes))
        
        self.query_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        self.key_weight = nn.Linear(input_channel, input_channel//factor, bias = False)
        
        self.use_adaptive_weighting = True
        self.epoch_count = 0
        self.prev_ensem_logits = {}
        self.current_epoch_ensem_logits = {}
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
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
        if self.epoch_count == 0 or sample_ids is None or len(self.prev_ensem_logits) == 0:
            batch_size = logitlist[0].size(0)
            num_branches = len(logitlist)
            uniform_weight = 1.0 / num_branches
            return [torch.ones(batch_size, device=logitlist[0].device) * uniform_weight for _ in range(num_branches)]
        
        batch_size = logitlist[0].size(0)
        device = logitlist[0].device
        num_branches = len(logitlist)
        num_classes = logitlist[0].size(1)
        
        prev_logits_batch, valid_mask = load_history_batch(
            self.prev_ensem_logits,
            sample_ids,
            batch_size,
            num_classes,
            device,
        )
        
        current_logits_stacked = torch.stack(logitlist, dim=0)
        
        if self.dissimilarity_metric == 'wasserstein1':
            current_probs = F.softmax(current_logits_stacked, dim=2)
            prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)
            
            current_cdfs = torch.cumsum(current_probs, dim=2)
            prev_cdfs = torch.cumsum(prev_probs, dim=2)
            
            dissim_dists = torch.sum(torch.abs(current_cdfs - prev_cdfs), dim=2)
            
        elif self.dissimilarity_metric == 'wasserstein2':
            current_probs = F.softmax(current_logits_stacked, dim=2)
            prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)
            
            current_cdfs = torch.cumsum(current_probs, dim=2)
            prev_cdfs = torch.cumsum(prev_probs, dim=2)
            
            dissim_dists = torch.sqrt(torch.sum((current_cdfs - prev_cdfs)**2, dim=2))
            
        elif self.dissimilarity_metric == 'euclidean':
            current_probs = F.softmax(current_logits_stacked, dim=2)
            prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)
            
            dissim_dists = torch.sqrt(torch.sum((current_probs - prev_probs)**2, dim=2))
            
        elif self.dissimilarity_metric == 'kl':
            current_probs = F.softmax(current_logits_stacked, dim=2)
            prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)
            
            eps = 1e-8
            current_probs = current_probs + eps
            prev_probs = prev_probs + eps
            
            dissim_dists = torch.sum(current_probs * torch.log(current_probs / prev_probs), dim=2)
            
        elif self.dissimilarity_metric == 'cosine':
            current_probs = F.softmax(current_logits_stacked, dim=2)
            prev_probs = F.softmax(prev_logits_batch.unsqueeze(0), dim=2)
            
            numerator = torch.sum(current_probs * prev_probs, dim=2)
            current_norm = torch.sqrt(torch.sum(current_probs**2, dim=2))
            prev_norm = torch.sqrt(torch.sum(prev_probs**2, dim=2))
            
            cosine_sim = numerator / (current_norm * prev_norm + 1e-8)
            dissim_dists = 1 - cosine_sim
            
        else:
            raise ValueError(f"Unknown dissimilarity metric: {self.dissimilarity_metric}")
        
        valid_mask_expanded = valid_mask.unsqueeze(0).expand(dissim_dists.shape[0], -1)
        dissim_dists = torch.where(valid_mask_expanded, dissim_dists, torch.ones_like(dissim_dists))
        
        dissim_matrix = dissim_dists.transpose(0, 1)
        
        normalized_dissim = F.softmax(dissim_matrix / self.tau, dim=1)
        
        return [normalized_dissim[:, i] for i in range(num_branches)]
    
    def update_epoch_history(self):
        self.prev_ensem_logits = self.current_epoch_ensem_logits.copy()
        self.current_epoch_ensem_logits = {}
        self.epoch_count += 1

    def record_epoch_logits(self, sample_ids, ensemble_logits):
        record_history_batch(
            self.current_epoch_ensem_logits, sample_ids, ensemble_logits
        )
        
    def forward(self, x, sample_ids=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = getattr(self,'layer3_0')(x)
        x_3 = self.avgpool(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        proj_q = self.query_weight(x_3)
        proj_q = proj_q[:, None, :]
        proj_k = self.key_weight(x_3)
        proj_k = proj_k[:, None, :]
        x_3_1 = getattr(self, 'classifier3_0')(x_3)
        pro = x_3_1.unsqueeze(-1) 
        if self.en:
            for i in range(1, self.num_branches):
                temp = getattr(self, 'layer3_'+str(i))(x)
                temp = self.avgpool(temp)
                temp = temp.view(temp.size(0), -1)   
                temp_q = self.query_weight(temp)
                temp_k = self.key_weight(temp)
                temp_q = temp_q[:, None, :]
                temp_k = temp_k[:, None, :]
                temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
                temp_1 = temp_1.unsqueeze(-1)
                pro = torch.cat([pro,temp_1],-1)
                proj_q = torch.cat([proj_q, temp_q], 1)
                proj_k = torch.cat([proj_k, temp_k], 1) 
            
            energy = torch.bmm(proj_q, proj_k.permute(0,2,1)) 
            attention = F.softmax(energy, dim = -1) 
            x_m = torch.bmm(pro, attention.permute(0,2,1))
            
            if self.use_adaptive_weighting:
                logitlist = [pro[:, :, i] for i in range(self.num_branches)]
                
                dissimilarities = self.compute_dissimilarities(logitlist, sample_ids)
                
                batch_size = pro.size(0)
                num_classes = pro.size(1)
                ensemble_logit = torch.zeros(batch_size, num_classes, device=pro.device)
                
                for i, (logit, weight) in enumerate(zip(logitlist, dissimilarities)):
                    weighted_logit = logit * weight.view(-1, 1)
                    ensemble_logit += weighted_logit
                
                return pro, x_m, ensemble_logit
            
            return pro, x_m
        else:
            for i in range(1, self.num_branches - 1):
                temp = getattr(self, 'layer3_'+str(i))(x)
                temp = self.avgpool(temp)
                temp = temp.view(temp.size(0), -1)   
                temp_q = self.query_weight(temp)
                temp_k = self.key_weight(temp)
                temp_q = temp_q[:, None, :]
                temp_k = temp_k[:, None, :]
                temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
                temp_1 = temp_1.unsqueeze(-1)
                pro = torch.cat([pro,temp_1],-1)
                proj_q = torch.cat([proj_q, temp_q], 1)
                proj_k = torch.cat([proj_k, temp_k], 1) 
            
            energy =  torch.bmm(proj_q, proj_k.permute(0,2,1)) 
            attention = F.softmax(energy, dim = -1) 
            x_m = torch.bmm(pro, attention.permute(0,2,1))
            
            temp = getattr(self, 'layer3_'+str(self.num_branches - 1))(x)
            temp = self.avgpool(temp)
            temp = temp.view(temp.size(0), -1)   
            temp_out = getattr(self, 'classifier3_' + str(self.num_branches - 1))(temp)
            
            if self.use_adaptive_weighting:
                logitlist = [pro[:, :, i] for i in range(self.num_branches - 1)]
                
                dissimilarities = self.compute_dissimilarities(logitlist, sample_ids)
                
                batch_size = pro.size(0)
                num_classes = pro.size(1)
                ensemble_logit = torch.zeros(batch_size, num_classes, device=pro.device)
                
                for i, (logit, weight) in enumerate(zip(logitlist, dissimilarities)):
                    weighted_logit = logit * weight.view(-1, 1)
                    ensemble_logit += weighted_logit
                
                return pro, x_m, temp_out, ensemble_logit
            
            return pro, x_m, temp_out
        
def resnet32(pretrained=False, path=None, **kwargs):
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def resnet110(pretrained=False, path=None, **kwargs):
    model = ResNet(Bottleneck, [12, 12, 12], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def wide_resnet20_8(pretrained=False, progress=True, **kwargs):
    model = ResNet(Bottleneck, [2, 2, 2], width_per_group = 64 * 8, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
