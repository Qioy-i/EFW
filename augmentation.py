import numpy as np
import torch
import random
import nibabel as nib
import torch.nn.functional as F
import torchio as tio

def augment_brightness_additive(data_sample, mu:float, sigma:float , per_channel:bool=True, p_per_channel:float=1.):
    """
    The input `data_sample` must have shape (c, x, y[, z]).
    
    :param data_sample: The input image sample.  
    :param mu: Mean of the random distribution, controlling the center of the brightness shift.  
    :param sigma: Standard deviation of the random distribution, determining the magnitude of brightness variation.  
    :param per_channel: Whether to apply the brightness shift independently per channel.  
    :param p_per_channel: Probability of applying the shift to each channel individually (if `per_channel` is True).  
    :return: Brightness-adjusted image sample.
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
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1]) # Randomly sample a value from [0.5, 2]; values < 1 darken the image, values > 1 brighten it.
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
        volume=tio.ScalarImage(tensor=volume) 
    )
    if p > 0.5:
        scale_factor = random.uniform(0.8, 1.2)

        zoom = tio.RandomAffine(
            scales=(scale_factor, scale_factor), 
            degrees=0,                            
            translation=0,                      
            isotropic=True                     
        )
        transformed_subject = zoom(subject)
        scale_factor_list = [scale_factor, scale_factor, scale_factor]
    else:
        scale_factor_list = np.random.uniform(0.8, 1.2, 3)

        zoom = tio.RandomAffine(
            scales=(
                scale_factor_list[0], scale_factor_list[0],
                scale_factor_list[1], scale_factor_list[1],
                scale_factor_list[2], scale_factor_list[2]
            ), 
            degrees=0, 
            translation=0,  
            isotropic=False 
        )
        transformed_subject = zoom(subject)
    
    volume_zoom = transformed_subject.volume.data

    return volume_zoom, scale_factor_list

def random_contrast_adjust(img, log_gamma=(-0.2, 0.2)):
    # (C, H, W, D)
    transform = tio.Compose([
        tio.RandomGamma(log_gamma), 
    ])

    p = torch.rand(1)
    if p > 0.5:
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
