import numpy as np
import torch
import random
import nibabel as nib
import torch.nn.functional as F
import torchio as tio

def augment_brightness_additive(data_sample, mu:float, sigma:float , per_channel:bool=True, p_per_channel:float=1.):
    """
    data_sample must have shape (c, x, y(, z)))
    :param data_sample: 随机数分布的均值，用来控制亮度偏移的中心值。
    :param mu:  随机数分布的标准差，决定亮度变化的幅度。
    :param sigma: 
    :param per_channel: 
    :param p_per_channel: 
    :return: 
    """
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                data_sample[c] += rnd_nb
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                data_sample[c] += rnd_nb
    return data_sample


def  augment_brightness_multiplicative(data_sample, multiplier_range=(0.6, 1.4), per_channel=True):
    # (C, H, W, D)
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1]) # 从0.5到2中随机采样，如果采样值<1,亮度变暗；如果采样值>1:亮度变量
    if not per_channel:
        data_sample *= multiplier
    else:
        for c in range(data_sample.shape[0]):
            multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
            data_sample[c] *= multiplier
    data_sample [data_sample>255] = 255
    return data_sample

def Rotate3D_volume(volume):
    # (H, W, D)
    
    angle_list = [0, 0, 0]
    p = torch.rand(1)
    if p > 0.5:
        pa = torch.rand(1)
        if (pa > 0.33)&(pa<=0.66):
            angle = np.random.uniform(-25, 25, 1)
            mode = torch.randint(3,(1,))
            angle_list[mode] = float(angle)
        elif pa > 0.66:
            angle = np.random.uniform(-25, 25, 2)
            mode = torch.randint(3,(2,))
            angle_list[mode[0]], angle_list[mode[1]] = float(angle[0]), float(angle[1])
        else:
            angle_list = np.random.uniform(-25, 25, 3)

    return volume

def random_zoom(volume):

    # (C, H, W, D)
    # p = torch.rand(1)
    p = 0.8
    # p = 0.1
    
    # 确保输入为 4D tensor (C, H, W, D)
    if len(volume.shape) == 3:  # (H, W, D)
        volume = volume.unsqueeze(0)  # (1, H, W, D)

    # 将 tensor 转换为 torchio 的 Subject
    subject = tio.Subject(
        volume=tio.ScalarImage(tensor=volume)  # 使用 ScalarImage 包装
    )
    if p > 0.5:
        scale_factor = random.uniform(0.8, 1.2)

        # 创建随机缩放变换，确保 isotropic 确保缩放因子对所有轴一致
        zoom = tio.RandomAffine(
            scales=(scale_factor, scale_factor),  # 将缩放因子传入
            degrees=0,                            # 无旋转
            translation=0,                        # 无平移
            isotropic=True                        # 确保所有轴缩放一致
        )
        # 应用变换
        transformed_subject = zoom(subject)
        scale_factor_list = [scale_factor, scale_factor, scale_factor]
    else:
        scale_factor_list = np.random.uniform(0.8, 1.2, 3)

        # 创建随机缩放变换，确保 isotropic 确保缩放因子对所有轴一致
        zoom = tio.RandomAffine(
            scales=(
                scale_factor_list[0], scale_factor_list[0],
                scale_factor_list[1], scale_factor_list[1],
                scale_factor_list[2], scale_factor_list[2]
            ),  # 每个轴的缩放范围 (min, max)
            degrees=0,  # 无旋转
            translation=0,  # 无平移
            isotropic=False  # 各轴独立缩放
        )
        transformed_subject = zoom(subject)
    
    # 从 Subject 中提取变换后的数据
    volume_zoom = transformed_subject.volume.data

    return volume_zoom, scale_factor_list

def random_contrast_adjust(img, log_gamma=(-0.2, 0.2)):
    # (C, H, W, D)
    transform = tio.Compose([
        tio.RandomGamma(log_gamma),  # 调整对比度
    ])

    p = torch.rand(1)
    if p > 0.5:
    # 应用变换
        img = img / 255.
        img = transform(img)
        img = img * 255.
    return img

def crop(img, crop_size):
    
    if isinstance(img, np.ndarray):
        img = torch.tensor(img)  
    if img.dim() == 3:  
        img = img.unsqueeze(0)  

    crop = tio.Crop((crop_size, crop_size, crop_size, crop_size, crop_size, crop_size))
    pad = tio.Pad((crop_size, crop_size, crop_size, crop_size, crop_size, crop_size))
    transform = tio.Compose([crop, pad])

    subject = tio.Subject(
        volume=tio.ScalarImage(tensor=img)
    )

    transformed_subject = transform(subject)
    processed_img = transformed_subject.volume.data  

    return processed_img.squeeze(0)
