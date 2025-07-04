import json
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import random
from augmentation import Rotate3D_volume, random_zoom, random_contrast_adjust, augment_brightness_multiplicative, crop
import time
import fnmatch
import bisect

class CustomDataset(Dataset):
    def __init__(self, root_dir, mode, body_parts, target_shape, spacing=[0.3, 0.3, 0.3], transform=None, is_aug = False):
        # 设置路径
        self.root_dir = os.path.join(root_dir, mode)
        self.body_parts = body_parts
        self.target_shape = target_shape
        self.spacing = torch.tensor(spacing, dtype=torch.float32)
        self.transform = transform
        self.is_aug = is_aug

        # 读取所有病例文件夹
        self.case_dirs = [os.path.join(self.root_dir, case) for case in os.listdir(self.root_dir)]
        self.abdomen_paths = []
        self.head_paths = []
        self.leg_paths = []
        self.labels = []
        self.days = []
        self.ACs = []
        self.HCs = []
        self.resize_ratio_head = []
        self.resize_ratio_abdomen = []
        self.resize_ratio_leg = []
        self.hadlocks = []
        self.mean = 82.53642591943878  # 头+腹
        self.std = 49.407943210983795


        # 读取每个病例的3D数据路径和金标准
        for case_dir in self.case_dirs:
            for file in os.listdir(case_dir):
                if file.endswith('.json'):
                    json_path = os.path.join(case_dir, file)  # JSON 文件名
                abdomen_path = os.path.join(case_dir, 'abdomen')
                head_path = os.path.join(case_dir, 'head')
                leg_path = os.path.join(case_dir, 'leg')
                abdomen = [os.path.join(abdomen_path, f) for f in os.listdir(abdomen_path) if fnmatch.fnmatch(f, f"*{'diffresize.vol.npy'}")][0]
                head = [os.path.join(head_path, f) for f in os.listdir(head_path) if fnmatch.fnmatch(f, f"*{'diffresize.vol.npy'}")][0] 
                leg = [os.path.join(leg_path, f) for f in os.listdir(leg_path) if fnmatch.fnmatch(f, f"*{'diffresize.vol.npy'}")][0] 

            with open(json_path, 'r') as f:
                details = json.load(f)

            AC = details['AC(mm)'] / 10
            HC = details['HC(mm)'] / 10
            FL = details['FL(mm)'] / 10
            hadlock3 = 10**(1.326-0.00326*AC*FL+0.0107*HC+0.0438*AC+0.158*FL)

            self.abdomen_paths.append(abdomen)
            self.head_paths.append(head)
            self.leg_paths.append(leg)
            self.labels.append(details['GroundTruth'])
            self.days.append(details['day(d)'])
            self.resize_ratio_head.append(details['head_resize_scale_factor_differnetresize'])
            self.resize_ratio_abdomen.append(details['abdomen_resize_scale_factor_differnetresize'])
            self.resize_ratio_leg.append(details['leg_resize_scale_factor_differnetresize'])
            self.hadlocks.append(hadlock3)

    def __len__(self):
        return len(self.abdomen_paths)

    def __getitem__(self, idx):
        abdomen_path = self.abdomen_paths[idx]
        resize_ratio_abdomen = self.resize_ratio_abdomen[idx]

        head_path = self.head_paths[idx]
        resize_ratio_head = self.resize_ratio_head[idx]

        leg_path = self.leg_paths[idx]
        resize_ratio_leg = self.resize_ratio_leg[idx]

        label = self.labels[idx] / 5000  
        new_label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        day = self.days[idx] / 10
        new_day = torch.tensor(day, dtype=torch.float32).unsqueeze(0)
        hadlock = self.hadlocks[idx] / 5000
        new_hadlock = torch.tensor(hadlock, dtype=torch.float32).unsqueeze(0)

        abdomen_data = np.load(abdomen_path)
        abdomen_data = torch.from_numpy(abdomen_data).float()

        head_data = np.load(head_path)
        head_data = torch.from_numpy(head_data).float()

        idx = torch.tensor(idx, dtype=torch.long)
        leg_data = np.load(leg_path)
        leg_data = torch.from_numpy(leg_data).float()

        if self.is_aug:
            abdomen_data1, abdomen_scale_factor1 = self.data_aug(abdomen_data)
            new_spacing_abdomen1 = self.spacing.clone()  
            new_spacing_abdomen1[0] = self.spacing[0] / resize_ratio_abdomen / abdomen_scale_factor1[0] * 0.2
            new_spacing_abdomen1[1] = self.spacing[1] / resize_ratio_abdomen / abdomen_scale_factor1[1] * 0.2
            new_spacing_abdomen1[2] = self.spacing[2] / resize_ratio_abdomen / abdomen_scale_factor1[2] * 0.2

            head_data1, head_scale_factor1 = self.data_aug(head_data)
            new_spacing_head1 = self.spacing.clone()  
            new_spacing_head1[0] = self.spacing[0] / resize_ratio_head / head_scale_factor1[0] * 0.2
            new_spacing_head1[1] = self.spacing[1] / resize_ratio_head / head_scale_factor1[1] * 0.2
            new_spacing_head1[2] = self.spacing[2] / resize_ratio_head / head_scale_factor1[2] * 0.2

            leg_data1, leg_scale_factor1 = self.data_aug(leg_data)
            new_spacing_leg1 = self.spacing.clone()  
            new_spacing_leg1[0] = self.spacing[0] / resize_ratio_leg / leg_scale_factor1[0] * 0.2
            new_spacing_leg1[1] = self.spacing[1] / resize_ratio_leg / leg_scale_factor1[1] * 0.2
            new_spacing_leg1[2] = self.spacing[2] / resize_ratio_leg / leg_scale_factor1[2] * 0.2

            abdomen_data2 = abdomen_data.unsqueeze(0)
            new_spacing_abdomen2 = self.spacing.clone()  
            new_spacing_abdomen2[0] = self.spacing[0] / resize_ratio_abdomen * 0.2
            new_spacing_abdomen2[1] = self.spacing[1] / resize_ratio_abdomen * 0.2
            new_spacing_abdomen2[2] = self.spacing[2] / resize_ratio_abdomen * 0.2

            head_data2 = head_data.unsqueeze(0)
            new_spacing_head2 = self.spacing.clone()  
            new_spacing_head2[0] = self.spacing[0] / resize_ratio_head * 0.2
            new_spacing_head2[1] = self.spacing[1] / resize_ratio_head * 0.2
            new_spacing_head2[2] = self.spacing[2] / resize_ratio_head * 0.2

            leg_data2 = leg_data.unsqueeze(0)
            new_spacing_leg2 = self.spacing.clone()  
            new_spacing_leg2[0] = self.spacing[0] / resize_ratio_leg * 0.2
            new_spacing_leg2[1] = self.spacing[1] / resize_ratio_leg * 0.2
            new_spacing_leg2[2] = self.spacing[2] / resize_ratio_leg * 0.2
        else:
            new_spacing_abdomen1 = self.spacing.clone()  
            new_spacing_abdomen1[0] = self.spacing[0] / resize_ratio_abdomen * 0.2
            new_spacing_abdomen1[1] = self.spacing[1] / resize_ratio_abdomen * 0.2
            new_spacing_abdomen1[2] = self.spacing[2] / resize_ratio_abdomen * 0.2

            new_spacing_abdomen2 = self.spacing.clone()  
            new_spacing_abdomen2[0] = self.spacing[0] / resize_ratio_abdomen * 0.2
            new_spacing_abdomen2[1] = self.spacing[1] / resize_ratio_abdomen * 0.2
            new_spacing_abdomen2[2] = self.spacing[2] / resize_ratio_abdomen * 0.2

            new_spacing_head1 = self.spacing.clone()  
            new_spacing_head1[0] = self.spacing[0] / resize_ratio_head * 0.2
            new_spacing_head1[1] = self.spacing[1] / resize_ratio_head * 0.2
            new_spacing_head1[2] = self.spacing[2] / resize_ratio_head * 0.2

            new_spacing_head2 = self.spacing.clone()  
            new_spacing_head2[0] = self.spacing[0] / resize_ratio_head * 0.2
            new_spacing_head2[1] = self.spacing[1] / resize_ratio_head * 0.2
            new_spacing_head2[2] = self.spacing[2] / resize_ratio_head * 0.2

            new_spacing_leg1 = self.spacing.clone()  
            new_spacing_leg1[0] = self.spacing[0] / resize_ratio_leg * 0.2
            new_spacing_leg1[1] = self.spacing[1] / resize_ratio_leg * 0.2
            new_spacing_leg1[2] = self.spacing[2] / resize_ratio_leg * 0.2

            new_spacing_leg2 = self.spacing.clone()  
            new_spacing_leg2[0] = self.spacing[0] / resize_ratio_leg * 0.2
            new_spacing_leg2[1] = self.spacing[1] / resize_ratio_leg * 0.2
            new_spacing_leg2[2] = self.spacing[2] / resize_ratio_leg * 0.2

            abdomen_data1 = abdomen_data.unsqueeze(0)
            head_data1 = head_data.unsqueeze(0)
            abdomen_data2 = abdomen_data.unsqueeze(0)
            head_data2 = head_data.unsqueeze(0)
            leg_data1 = leg_data.unsqueeze(0)
            leg_data2 = leg_data.unsqueeze(0)

        abdomen_data1 = self.__normalize_intensity(abdomen_data1, self.mean, self.std)
        head_data1 = self.__normalize_intensity(head_data1, self.mean, self.std)
        abdomen_data2 = self.__normalize_intensity(abdomen_data2, self.mean, self.std)
        head_data2 = self.__normalize_intensity(head_data2, self.mean, self.std)   
        leg_data1 = self.__normalize_intensity(leg_data1, self.mean, self.std)  
        leg_data2 = self.__normalize_intensity(leg_data2, self.mean, self.std)  

        return {
        'abdomen_data1': abdomen_data1,
        'abdomen_data2': abdomen_data2,
        'head_data1': head_data1,
        'head_data2': head_data2,
        'leg_data1': leg_data1,
        'leg_data2': leg_data2,
        'total_label': new_label,
        'abdomen_spacing1': new_spacing_abdomen1,
        'abdomen_spacing2': new_spacing_abdomen2,
        'head_spacing1': new_spacing_head1,
        'head_spacing2': new_spacing_head2,
        'leg_spacing1': new_spacing_leg1,
        'leg_spacing2': new_spacing_leg2,
        'idx':idx,
        'day': new_day,
        'hadlock': new_hadlock,
        'paths': (abdomen_path, head_path)
    }

    def data_aug(self, volume):
        # 旋转翻转
        volume = self.rotate_flip_volume(volume)   # 3D
        # 任意角度旋转
        volume = Rotate3D_volume(volume)   # 3D  25°

        volume = volume.unsqueeze(0)
        # 亮度
        volume = augment_brightness_multiplicative(volume)   # 4D
        # 对比度
        volume = random_contrast_adjust(volume)    # 4D
        # 随机缩放
        volume, scale_factor = random_zoom(volume)   # 4D


        return volume, scale_factor



    def __normalize_intensity(self, volume, mean, std):
        """对体积数据进行归一化"""
        volume = (volume - mean) / (std + 1e-5)
        return volume

    def resize_img(self, volume, fixsize):
        ow,oh,od = volume.shape[-3:]
        w,h,d = fixsize
        volume = volume.unsqueeze(0).unsqueeze(0)  # 添加批次维度
        resize_ratio = min(w/ow, h/oh, d/od)
        des_img = F.interpolate(volume, scale_factor=resize_ratio, mode='trilinear', align_corners = True)
        nw, nh, nd = int(ow*resize_ratio), int(oh*resize_ratio), int(od*resize_ratio)
        des_img = F.pad(des_img,pad=(0,d-nd,0,h-nh,0,w-nw),mode="constant", value = 0)
        des_img = des_img.squeeze()
        return des_img, resize_ratio


    def rotate_pi_along_axis(self, volume, axis):
        # 检查 axis 是否有效
        assert axis in [0, 1, 2], "Axis must be 0 , 1 , or 2."
        assert isinstance(volume, torch.Tensor), "Volume must be a torch.Tensor."

        # # 根据旋转的轴，翻转其他两个轴
        if axis == 0:  
            rotated_volume = torch.rot90(volume, 2, [1, 2])   
        elif axis == 1:  
            rotated_volume = torch.rot90(volume, 2, [0, 2])   
        elif axis == 2:  
            rotated_volume = torch.rot90(volume, 2, [0, 1])

        return rotated_volume


    def flip_along_axis(self, volume, axis):
        # 检查 axis 是否有效
        assert axis in [0, 1, 2], "Axis must be 0 , 1 , or 2 ."
        assert isinstance(volume, torch.Tensor), "Volume must be a torch.Tensor."

        # 使用 torch.flip 沿指定轴翻转张量
        flipped_volume = torch.flip(volume, dims=[axis])
        return flipped_volume

    def flip_volume(self, volume):
        if len(volume.shape) > 3:
            volume = volume.squeeze()

        p1 = np.random.uniform(0,1)                 
        p2 = np.random.uniform(0,1) 

        if p1 > 0.5:
            torch.flip(volume, dims=[0])

        if p2 > 0.5:
            torch.flip(volume, dims=[1])

        return volume


    def rotate_flip_volume(self, volume):
        if len(volume.shape) > 3:
            volume.squeeze()

        p1 = np.random.uniform(0,1)                 
        p2 = np.random.uniform(0,1) 
        axis1 = random.choice([0, 1, 2])      
        axis2 = random.choice([0, 1, 2])      
        if p1 > 0.5:
            volume = self.rotate_pi_along_axis(volume, axis1)            
        if p2 > 0.5:
            volume = self.flip_along_axis(volume, axis2)

        return volume
   
# 修改获取训练数据的函数
def get_training_data(root_dir, body_parts, target_shape, transform=None, is_aug=True):
    dataset = CustomDataset(root_dir, "train", body_parts, target_shape, transform=transform, is_aug = is_aug)
    return dataset

def get_validation_data(root_dir, body_parts, target_shape, transform=None):
    dataset = CustomDataset(root_dir, "val", body_parts, target_shape, transform=transform)
    return dataset

def get_test_data(root_dir, body_parts, target_shape, transform=None):
    dataset = CustomDataset(root_dir, "test", body_parts, target_shape, transform=transform)
    return dataset
