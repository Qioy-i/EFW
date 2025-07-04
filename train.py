import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import get_training_data, get_validation_data
from tqdm import tqdm
from model import model
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from warmupcosineLR import WarmupCosineLR
import random
import math
torch.cuda.empty_cache()
torch.cuda.set_device(1)
import time

import warnings
warnings.filterwarnings("ignore", message="Using TorchIO images without a torchio.SubjectsLoader*")

def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_collate_fn(batch):
    
    abdomen_data1 = torch.stack([item['abdomen_data1'] for item in batch], dim=0)
    head_data1 = torch.stack([item['head_data1'] for item in batch], dim=0)
    leg_data1 = torch.stack([item['leg_data1'] for item in batch], dim=0)
    abdomen_data2 = torch.stack([item['abdomen_data2'] for item in batch], dim=0)
    head_data2 = torch.stack([item['head_data2'] for item in batch], dim=0)
    leg_data2 = torch.stack([item['leg_data2'] for item in batch], dim=0)
    total_label = torch.stack([item['total_label'] for item in batch], dim=0)
    abdomen_spacing1 = torch.stack([item['abdomen_spacing1'] for item in batch], dim=0)
    head_spacing1 = torch.stack([item['head_spacing1'] for item in batch], dim=0)
    leg_spacing1 = torch.stack([item['leg_spacing1'] for item in batch], dim=0)
    abdomen_spacing2 = torch.stack([item['abdomen_spacing2'] for item in batch], dim=0)
    head_spacing2 = torch.stack([item['head_spacing2'] for item in batch], dim=0)
    leg_spacing2 = torch.stack([item['leg_spacing2'] for item in batch], dim=0)
    days = torch.stack([item['day'] for item in batch], dim=0)
    hadlocks = torch.stack([item['hadlock'] for item in batch], dim=0)
    paths = [item['paths'] for item in batch]
    
    combined_data1 = torch.cat([head_data1, abdomen_data1 ], dim=0) 
    combined_spacing1 = torch.cat([head_spacing1, abdomen_spacing1 ], dim=0)
    combined_data2 = torch.cat([head_data2, abdomen_data2 ], dim=0) 
    combined_spacing2 = torch.cat([head_spacing2, abdomen_spacing2 ], dim=0)
    combined_days = torch.cat([days, days], dim=0)
    return combined_data1, combined_data2, total_label, combined_spacing1, combined_spacing2, combined_days, hadlocks, paths

def load_hyperparameters(hyperparameters_path):
    hyperparameters = torch.load(hyperparameters_path)
    return hyperparameters


def compute_mae(predictions, labels):
    return np.mean(np.abs(predictions - labels))

def compute_rmse(predictions, labels):
    return np.sqrt(np.mean((predictions - labels) ** 2))

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(model, data_loader, criterion, optimizer, device, epochs):
    model.train()
    total_loss = 0

    weight_pred_arr = []
    weight_gt_arr = []

    for i, (data1, data2,  total_labels, spacing1, spacing2,  day, hadlock, path) in enumerate(tqdm(data_loader, desc="Training")):
        data1 = data1.to(device)
        data2 = data2.to(device)
        labels = total_labels.to(device)
        spacing1 = spacing1.to(device)
        spacing2 = spacing2.to(device)
        day = day.to(device)

        fw_gt = labels
    
        optimizer.zero_grad()
        epochs[2] += 1
        loss, outputs = model(data1, data2, spacing1, spacing2, fw_gt, day, epochs)
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
        
        total_loss += loss.item()

        weight_pred_arr.extend(outputs.view(-1).cpu().detach().numpy() * 5000)
        weight_gt_arr.extend(fw_gt.view(-1).cpu().detach().numpy() * 5000)

    weight_mae = compute_mae(np.array(weight_pred_arr), np.array(weight_gt_arr))
    weight_rmse = compute_rmse(np.array(weight_pred_arr), np.array(weight_gt_arr))

    return total_loss / len(data_loader), weight_mae, weight_rmse

def validate(model, data_loader, criterion, device, epochs):
    model.eval()
    total_loss = 0

    weight_pred_arr = []
    weight_gt_arr = []

    for i, (data1, data2,  total_labels, spacing1, spacing2,  day, hadlock, path) in enumerate(tqdm(data_loader, desc="Validation")):
        data1 = data1.to(device)
        data2 = data2.to(device)
        labels = total_labels.to(device)
        spacing1 = spacing1.to(device)
        spacing2 = spacing2.to(device)
        day = day.to(device)

        fw_gt = labels

        with torch.no_grad():
            loss, outputs = model(data1, data2, spacing1, spacing2, fw_gt, day, epochs)
        
            total_loss += loss.item()

        weight_pred_arr.extend(outputs.view(-1).cpu().detach().numpy() * 5000)
        weight_gt_arr.extend(fw_gt.view(-1).cpu().detach().numpy() * 5000)

    weight_mae = compute_mae(np.array(weight_pred_arr), np.array(weight_gt_arr))
    weight_rmse = compute_rmse(np.array(weight_pred_arr), np.array(weight_gt_arr))

    return total_loss / len(data_loader), weight_mae, weight_rmse

def save_model(weights_dir, model, epoch, text):
    path = os.path.join(weights_dir, f'{text}_{epoch}.pth')
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")

def save_results(epoch, results, save_dir):
    results_converted = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in results.items()}
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, 'a') as f:
        json.dump({f"epoch_{epoch}": results_converted}, f)
        f.write('\n')


def plot_metrics(body_parts, save_dir, total_metrics, epoch):
    epochs = range(1, epoch + 1)
    plt.figure(figsize=(18, 5))

    metrics = ["Loss", "MAE", "RMSE"]
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.plot(epochs, total_metrics[i - 1][0], label=f'Train {metric}', color = 'blue')
        plt.plot(epochs, total_metrics[i - 1][1], label=f'Val {metric}', color = 'red')
        plt.title(f'Train_Val {metric} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()

    plt.suptitle(f'{body_parts} Training and Val Metrics after Epoch {epoch}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'train_val_metrics_epoch_{body_parts}.png'))
    plt.close()

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_hyperparameters(epoch, model_state_dict, optimizer, criterion, learning_rate, model_depth, 
                         save_dir, lr_scheduler_type, batch_size, target_shape, scheduler_state):

    hyperparameters = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_type": optimizer.__class__.__name__,
        "loss_function": criterion.__class__.__name__,
        "learning_rate": learning_rate,
        "model_depth": model_depth,
        "batch_size": batch_size,
        "target_shape": target_shape,
        "lr_scheduler_type": lr_scheduler_type,
        "scheduler_state_dict": scheduler_state
    }
    os.makedirs(save_dir, exist_ok=True)
    hyperparameters_path = os.path.join(save_dir, "hyperparameters.pt")

    torch.save(hyperparameters, hyperparameters_path)


def main():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--root_path', type=str, default=f'/dataset')
    parser.add_argument('--body_parts', type=str, default='head abdomen', help="Name of the body part")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--target_shape', type=list, default=[160, 128, 96])
    parser.add_argument('--load_hyperparams', type=bool, default=False)
    parser.add_argument('--hyperparams_path', type=str, default=r'')
    args = parser.parse_args()
    
    set_seed(3407)

    base_path = r'your save path'
    save_dir = os.path.join(base_path, "results")
    weights_dir = os.path.join(base_path, "pth")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    model_depth = 18
    model = MeanTeacher(alpha_min=0.99, alpha_max=0.9999, gamma_min=0, gamma_max=0.1, lambda_=0.001).to(device)

    if args.load_hyperparams:

        loaded_hyperparams = load_hyperparameters(args.hyperparams_path)
        model.load_state_dict(loaded_hyperparams['model_state_dict'])
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(loaded_hyperparams['learning_rate']))
        optimizer.load_state_dict(loaded_hyperparams['optimizer_state_dict'])

        scheduler = WarmupCosineLR(optimizer, lr_min= 1e-9, lr_max=args.learning_rate, warm_up=5, T_max=args.num_epochs)
        scheduler.load_state_dict(loaded_hyperparams['scheduler_state_dict'])
        start_epoch = loaded_hyperparams['epoch'] + 1
        batch_size = loaded_hyperparams['batch_size']
        print(f"Starting training from epoch {start_epoch}, batch size {batch_size}")

    else:
        
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

        scheduler = WarmupCosineLR(optimizer, lr_min= 1e-9, lr_max=args.learning_rate, warm_up=5, T_max=args.num_epochs)
        

    criterion = nn.MSELoss()

    transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x).float()), 
])
    train_dataset = get_training_data(args.root_path, args.body_parts, args.target_shape, transform)
    val_dataset = get_validation_data(args.root_path, args.body_parts, args.target_shape, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False, collate_fn=custom_collate_fn)

    train_losses, val_losses = [], []
    train_w_mae, val_w_mae = [], []
    train_w_rmse, val_w_rmse = [], []
    min_val_loss = float('inf')
    min_val_mae = float('inf')

    steps = math.ceil(len(train_dataset)/args.batch_size) * args.num_epochs
    step = 0
    for epoch in range(1, args.num_epochs + 1):
        step += (epoch-1) * math.ceil(len(train_dataset)/args.batch_size)
        epochs = [epoch, args.num_epochs, step, steps]
        train_output = train_one_epoch(model, train_loader, criterion, optimizer, device, epochs)
        val_output = validate(model, val_loader, criterion, device, epochs)

        train_losses.append(train_output[0])
        train_w_mae.append(train_output[1])
        train_w_rmse.append(train_output[2])
      
        val_losses.append(val_output[0])
        val_w_mae.append(val_output[1])
        val_w_rmse.append(val_output[2])
      
        if val_output[0] < min_val_loss:
            min_val_loss = val_output[0]
            save_model(weights_dir, model, epoch, 'best_loss')

        scheduler.step()
        current_lr = get_current_lr(optimizer)

        results = {
            "train_loss": train_output[0],
            "train_w_mae": train_output[1],
            "train_w_rmse": train_output[2],
            "val_loss": val_output[0],
            "val_w_mae": val_output[1],
            "val_w_rmse": val_output[2],
            "learning_rate": f"{current_lr:.8f}"
        }
        save_results(epoch, results, save_dir)

        save_hyperparameters(epoch, model.state_dict(), optimizer, criterion, args.learning_rate, model_depth, save_dir, "WarmupCosineLR", args.batch_size, args.target_shape, scheduler.state_dict())
        
        plot_metrics('weight', save_dir, [[train_losses,val_losses], [train_w_mae,val_w_mae], [train_w_rmse,val_w_rmse]], epoch)
        print(datetime.now().strftime("%Y%m%d-%H:%M:%S"))
        print(f"Epoch [{epoch}/{args.num_epochs}] Train Loss: {train_output[0]:.4f}, Val Loss: {val_output[0]:.8f}, Learning Rate: {current_lr:.8f}, Train w MAE: {train_output[1]:.4f}, Val w MAE: {val_output[1]:.4f}")


if __name__ == "__main__":
    main()
