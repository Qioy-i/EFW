import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from mamba_ssm import Mamba
from typing import Callable
from torchvision.ops.misc import MLP



def load_pretrained_weights(model, checkpoint_path):
    print(f"Loading pre-trained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'),weights_only=True)
  
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape} 
    
    print(f"Successfully loaded {len(pretrained_dict)} layers from pre-trained weights.")
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    return model


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Model(nn.Module):
    def __init__(self,
                 alpha_min = 0.9,
                 alpha_max = 0.995,
                 gamma_max = 0.5,
                 gamma_min = 0.0,
                 lambda_ = 0.0001):
        super().__init__()   
        self.alpha_max =  alpha_max
        self.alpha_min =  alpha_min
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.lambda_ = lambda_

        self.student_model = generate_model_spacing(model_depth=18)
        pretrained_weights_path = "/pth/r3d18_K_200ep.pth" 
        self.student_model = load_pretrained_weights(self.student_model, pretrained_weights_path)
        self.teacher_model = copy.deepcopy(self.student_model)


    def  ema(self, step, steps):
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) / 2 * (1 - np.cos(np.pi * step / steps))
        with torch.no_grad():
            student_state_dict = self.student_model.state_dict()
            teacher_state_dict = self.teacher_model.state_dict()
            for entry in teacher_state_dict.keys():
                teacher_param = teacher_state_dict[entry].clone().detach()
                student_param = student_state_dict[entry].clone().detach()
                new_param = (teacher_param * alpha) + (student_param * (1. - alpha))
                teacher_state_dict[entry] = new_param
            self.teacher_model.load_state_dict(teacher_state_dict)

    def rank_loss(self, x, g):

        x1 = x.repeat_interleave(x.shape[0], dim=0) 
        x2 = x.repeat(x.shape[0], 1)  
        diff_x = x1 - x2.detach()

        g1 = g.repeat_interleave(g.shape[0], dim=0)  
        g2 = g.repeat(g.shape[0], 1) 
        diff_g = g1 - g2

        idx_pos = diff_g > 0
        idx_neg = diff_g < 0

        diff_x_pos = diff_x[idx_pos]
        diff_x_neg = diff_x[idx_neg]


        if diff_x_pos.shape[0] == 0:
            loss = 0
            return loss
            
        else:
            diff_pos = torch.zeros((diff_x_pos.shape[0],2), device=x.device)
            diff_neg = torch.zeros((diff_x_neg.shape[0],2), device=x.device)

            diff_pos[:, 0] = - diff_x_pos
            diff_neg[:, 0] = diff_x_neg

            loss_pos = torch.max(diff_pos, dim=1)[0]
            loss_neg = torch.max(diff_neg, dim=1)[0]

            loss = torch.cat([loss_pos, loss_neg], dim=0)

            return loss.mean()

  #  inference
    def forward(self, x1, spacing1, day):
        out_s1 = self.student_model(x1, spacing1, day)  
        out_s1 = torch.sigmoid(out_s1)
        
        with torch.no_grad():
            out_t1 = self.teacher_model(x1, spacing1, day)
            out_t1 = torch.sigmoid(out_t1)

        return out_s1 , out_t1

  #  training
    def forward(self, x1, x2, spacing1, spacing2, gt_fw, day, epochs):
        self.ema(step = epochs[2], steps = epochs[3])

        out_s = self.student_model(x1, spacing1, day)  # B**2
        
        with torch.no_grad():
            out_t = self.teacher_model(x2, spacing2, day)

        out_s = torch.sigmoid(out_s)
        out_t = torch.sigmoid(out_t)

        mask_gt = torch.eye(int(x1.shape[0]/2), dtype=torch.bool, device = out_s.device).flatten()

        loss_w = nn.functional.mse_loss(out_s[mask_gt], gt_fw)

        loss_rank_s = self.rank_loss(out_s[mask_gt], gt_fw)

        gamma = self.gamma_min + (self.gamma_max- self.gamma_min) * epochs[0] / epochs[1]

        if out_s.shape[0] == 1:
            loss_fw = 0
        else:
            loss_fw = nn.functional.mse_loss(out_s[~mask_gt], out_t[~mask_gt])

        loss = loss_w + self.lambda_* loss_rank_s + loss_fw  * gamma
        
        return loss, out_t[mask_gt]



class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class MultiEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        # Attention block

        self.mamba1 = Mamba(d_model = hidden_dim)
        self.mamba2 = Mamba(d_model = hidden_dim)
        self.mamba3 = Mamba(d_model = hidden_dim)
        self.mamba4 = Mamba(d_model = hidden_dim)
        self.mamba5 = Mamba(d_model = hidden_dim)
        self.mamba6 = Mamba(d_model = hidden_dim)


        self.fc6 = nn.Sequential(
            nn.Linear(360, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )


    
    def forward(self, input: torch.Tensor, model = 'w6'):
        torch._assert(input.dim() == 5, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        B,C,D,H,W = input.shape

        x1 = input.view(B, C, -1).permute(0, 2, 1)                      
        x2 = torch.transpose(input, dim0=3, dim1=4).reshape(B, C, -1).permute(0, 2, 1)    
        x3 = torch.rot90(input,1, dims=(2,4)).reshape(B, C, -1).permute(0, 2, 1)        

        x4 = torch.flip(x1, dims=[1])
        x5 = torch.flip(x2, dims=[1])
        x6 = torch.flip(x3, dims=[1])

        x1 = self.mamba1(x1)  
        x2 = self.mamba2(x2)  
        x3 = self.mamba3(x3)   
        x4 = self.mamba4(x4)   
        x5 = self.mamba5(x5)   
        x6 = self.mamba6(x6)  

        x1 = x1.permute(0,2,1).reshape(B,C,D,H,W).view(B, C, -1)  
        x2 = x2.permute(0,2,1).reshape(B,C,D,W,H).transpose(dim0=3, dim1=4).reshape(B, C, -1)
        x3 = x3.permute(0,2,1).reshape(B,C,W,H,D).rot90(-1, dims=(2,4)).reshape(B, C, -1)

        x4 = x4.permute(0,2,1).flip(dims=[2]).reshape(B,C,D,H,W).view(B, C, -1)
        x5 = x5.permute(0,2,1).flip(dims=[2]).reshape(B,C,D,W,H).transpose(dim0=3, dim1=4).reshape(B, C, -1)
        x6 = x6.permute(0,2,1).flip(dims=[2]).reshape(B,C,W,H,D).rot90(-1, dims=(2,4)).reshape(B, C, -1)

        x = (x1+x2+x3+x4+x5+x6)/6   

        return x


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels + 3,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self. maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        
        self.fpn_lateral4 = nn.Conv3d(block_inplanes[3] * block.expansion, 128, kernel_size=1)
        self.fpn_lateral3 = nn.Conv3d(block_inplanes[2] * block.expansion, 128, kernel_size=1)
        self.fpn_lateral2 = nn.Conv3d(block_inplanes[1] * block.expansion, 128, kernel_size=1)
        self.fpn_lateral1 = nn.Conv3d(block_inplanes[0] * block.expansion, 128, kernel_size=1)

        self.fpn_out1 = nn.Conv3d(129, 128, kernel_size=8, padding=0, stride=8)
        self.fpn_out2 = nn.Conv3d(129, 128, kernel_size=4, padding=0, stride=4)
        self.fpn_out3 = nn.Conv3d(129, 128, kernel_size=2, padding=0, stride=2)
        self.fpn_out4 = nn.Conv3d(129, 128, kernel_size=1, padding=0, stride=1)

        self.conv1x1 = nn.Conv3d(in_channels=block_inplanes[3] * block.expansion * 2, out_channels=2048, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  

        self.spa_query = MultiEncoderBlock(hidden_dim=512,
                                           mlp_dim=1024,
                                           dropout=0.0,
                                           attention_dropout=0.0)

        self.ah_fc = nn.Sequential(
            nn.Linear(1024, 1)
        ) 

        self.channel_weight4 = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        ) 

        self.channel_weight512 = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 512)
        ) 

        self.spatial_weight60 = nn.Sequential(
            nn.Linear(60, 8),
            nn.ReLU(),
            nn.Linear(8, 60)
        ) 
      
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)


    def cnn(self, x, spacing):
        spacing =  spacing.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 160, 128, 96)
        x = torch.cat([x, spacing], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

    def fd_cat(self, x1, x2, x3, x4, day):
        f1 = self.fpn_lateral1(x1)
        d1 = torch.ones_like(f1,device=f1.device)[:,0,].unsqueeze(1) * day.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 40, 32, 24)
        f1 = torch.cat([f1, d1], dim=1)

        f2 = self.fpn_lateral2(x2)
        d2 = torch.ones_like(f2,device=f2.device)[:,0,].unsqueeze(1) * day.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 20, 16, 12)
        f2 = torch.cat([f2, d2], dim=1)

        f3 = self.fpn_lateral3(x3)
        d3 = torch.ones_like(f3,device=f3.device)[:,0,].unsqueeze(1) * day.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 10, 8, 6)
        f3 = torch.cat([f3, d3], dim=1)

        f4 = self.fpn_lateral4(x4)
        d4 = torch.ones_like(f4,device=f4.device)[:,0,].unsqueeze(1) * day.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, 5, 4, 3)
        f4 = torch.cat([f4, d4], dim=1)
        
        return f1, f2, f3, f4

    def down(self, f1, f2, f3, f4):
        f1 = self.fpn_out1(f1)
        f2 = self.fpn_out2(f2)
        f3 = self.fpn_out3(f3)
        f4 = self.fpn_out4(f4)

        return f1, f2, f3, f4
    
    def channel_weighting(self, x, all_channel = False):

        B,C,D,H,W = x.shape


        x_ = x.view(B,C,-1)  
        x_ = torch.mean(x_, dim= -1) 

        if all_channel:
            w = self.channel_weight512(x_)
        else:
            w = self.channel_weight4(x_) 
            w = w.repeat_interleave(C // 4, dim=1) 

        w = torch.sigmoid(w) * 2      
        x = x * w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        return x 


    def forward(self, x, spacing, day):

        x1, x2, x3, x4 = self.cnn(x, spacing)
        f1, f2, f3, f4 = self.fd_cat(x1, x2, x3, x4, day)
        f1, f2, f3, f4 = self.down(f1, f2, f3, f4)

        x = torch.cat([f1, f2, f3, f4], dim=1) 
        x = self.channel_weighting(x) 

        x = self.spa_query(x)  

        x = x.view(x.shape[0], x.shape[1], -1) 
        x = torch.mean(x, dim=-1)


        head = x[:int(x.shape[0]/2), :]   
        abdomen = x[int(x.shape[0]/2):, :]
        
        head_ = head.repeat(head.shape[0], 1)  
        abdomen_ = abdomen.repeat_interleave(abdomen.shape[0], dim=0) 
        ah = torch.cat([head_, abdomen_], dim = 1)

        x = self.ah_fc(ah)
    
        return x


def generate_model_spacing(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model
