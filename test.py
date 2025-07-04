import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import get_test_data,get_validation_data, get_training_data
from model import model
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import shutil
import scipy.stats as stats
import glob
from scipy.stats import linregress

torch.cuda.set_device(7)

def plot_bland_altman(label_data, predict_data, save_path, text, y_min=0, y_max=0):
    means = [(l + p) / 2 for l, p in zip(label_data, predict_data)]
    differences = [l - p for l, p in zip(label_data, predict_data)]
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    lower_limit = mean_diff - 1.96 * std_diff
    upper_limit = mean_diff + 1.96 * std_diff

    plt.figure(figsize=(10, 6))
    plt.scatter(means, differences, alpha=0.5)
    plt.axhline(mean_diff, color='gray', linestyle='--')
    plt.axhline(lower_limit, color='red', linestyle='--')
    plt.axhline(upper_limit, color='red', linestyle='--')
    plt.xlabel('Mean Error(g)',fontsize=20)
    plt.ylabel('Errors between AI and GT(g)',fontsize=20)
    plt.title(text,fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(save_path, f'{text}.png'))

def plot_consistency(label_data, predict_data, save_path, text):
    plt.figure(figsize=(10, 6))
    plt.scatter(label_data, predict_data, alpha=0.5, label='Predictions')
    slope, intercept, r_value, p_value, std_err = linregress(label_data, predict_data)
    regression_line = [slope * x + intercept for x in label_data]
    
    min_val = min(min(label_data), min(predict_data))
    max_val = max(max(label_data), max(predict_data))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', label='Ideal Consistency')
    
    plt.plot(label_data, regression_line, linestyle='-', color='red', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    
    plt.xlabel('Ground Truth (g)', fontsize=20)
    plt.ylabel('AI Predictions (g)', fontsize=20)
    plt.title(text, fontsize=20)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig(os.path.join(save_path, f'{text}.png'))

def save_predictions_to_json(pred_mem, t_pred_mem, save_dir):
    save_path = os.path.join(save_dir, 'output.json')
    data = {
        'pred_mem': pred_mem.tolist(),  
        't_pred_mem': t_pred_mem.tolist() 
    }

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def custom_collate_fn(batch):
    abdomen_data = torch.stack([item['abdomen_data1'] for item in batch], dim=0)
    head_data = torch.stack([item['head_data1'] for item in batch], dim=0)
    leg_data = torch.stack([item['leg_data1'] for item in batch], dim=0)
    total_label = torch.stack([item['total_label'] for item in batch], dim=0)
    abdomen_spacing = torch.stack([item['abdomen_spacing1'] for item in batch], dim=0)
    head_spacing = torch.stack([item['head_spacing1'] for item in batch], dim=0)
    leg_spacing = torch.stack([item['leg_spacing1'] for item in batch], dim=0)
    days = torch.stack([item['day'] for item in batch], dim=0)
    hadlocks = torch.stack([item['hadlock'] for item in batch], dim=0)
    paths = [item['paths'] for item in batch]
    
    combined_data = torch.cat([head_data, abdomen_data], dim=0)  
    combined_spacing = torch.cat([head_spacing, abdomen_spacing], dim=0)
    combined_days = torch.cat([days, days], dim=0)

    return combined_data, total_label, combined_spacing, combined_days, sln, hadlocks, paths

def pearsonr_pval(x, y):
    r, p = stats.pearsonr(x, y)
    return r, p

def compute_mae(predictions, labels):
    return np.mean(np.abs(predictions - labels))

def compute_rmse(predictions, labels):
    return np.sqrt(np.mean((predictions - labels) ** 2))

def compute_mape(predictions, labels):
    return np.mean(np.abs((predictions - labels) / labels)) * 100

def compute_sd(errors):
    return np.std(errors)

def calculate_percentage_within_threshold(errors, threshold):
    return np.mean(np.abs(errors) <= threshold) * 100

def test(model, data_loader, device, save_dir):
    model.eval()

    num = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    pred_mem = torch.zeros(num, 1).to(device)
    t_pred_mem = torch.zeros(num, 1).to(device)
    gt_mem = torch.zeros(num, 1).to(device)
    hadlocks_mem = torch.zeros(num, 1).to(device)

    error_paths = []
    large_error_data = [] 

    s_small_example = []
    small_gt = []
    s_large_example = []
    large_gt = []
    small_hadlock = []
    large_hadlock = []
    s_normal_example = []
    normal_gt = []
    normal_hadlock = []

    t_small_example = []
    t_large_example = []
    t_normal_example = []

    with torch.no_grad():
        for i, (data, labels, spacing, day, sln, hadlock, path) in enumerate(tqdm(data_loader, desc="Testing", leave=True)):
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            data, labels = data.to(device).float(), labels.to(device)
            day = day.to(device)
            spacing = spacing.to(device)
            
            out_s1, out_t1 = model(data, spacing, day)

            hadlock = hadlock.to(device)

            pred_mem[i*batch_size: i*batch_size + int(data.size(0)/2)] = out_s1.unsqueeze(-1) * 5000
            gt_mem[i*batch_size: i*batch_size + int(data.size(0)/2)] = labels[:,0].unsqueeze(-1) * 5000
            hadlocks_mem[i*batch_size: i*batch_size + int(data.size(0)/2)] = hadlock * 5000

            t_pred_mem[i*batch_size: i*batch_size + int(data.size(0)/2)] = out_t1.unsqueeze(-1) * 5000
            error_paths.extend(path)

    pred_mem = pred_mem.view(-1).cpu().detach().numpy()
    t_pred_mem = t_pred_mem.view(-1).cpu().detach().numpy()
    gt_mem = gt_mem.view(-1).cpu().detach().numpy()
    hadlocks_mem = hadlocks_mem.view(-1).cpu().detach().numpy()
    errors = pred_mem - gt_mem
    t_errors = t_pred_mem - gt_mem

    for i in range(len(gt_mem)):
        if gt_mem[i]<2500:
            s_small_example.append(pred_mem[i])
            t_small_example.append(t_pred_mem[i])
            small_gt.append(gt_mem[i])
            small_hadlock.append(hadlocks_mem[i])
        elif gt_mem[i]>4000:
            s_large_example.append(pred_mem[i])
            t_large_example.append(t_pred_mem[i])
            large_gt.append(gt_mem[i])
            large_hadlock.append(hadlocks_mem[i])
        else:
            s_normal_example.append(pred_mem[i])
            t_normal_example.append(t_pred_mem[i])
            normal_gt.append(gt_mem[i])
            normal_hadlock.append(hadlocks_mem[i])

    s_small_example = np.array(s_small_example)
    t_small_example = np.array(t_small_example)
    small_gt = np.array(small_gt)
    small_hadlock = np.array(small_hadlock)

    s_large_example = np.array(s_large_example)
    t_large_example = np.array(t_large_example)
    large_gt = np.array(large_gt)
    large_hadlock = np.array(large_hadlock)

    s_normal_example = np.array(s_normal_example)
    t_normal_example = np.array(t_normal_example)
    normal_gt = np.array(normal_gt)
    normal_hadlock = np.array(normal_hadlock)

    s_small_error = np.array(s_small_example)-np.array(small_gt)
    s_large_error = np.array(s_large_example)-np.array(large_gt)
    s_normal_error = np.array(s_normal_example)-np.array(normal_gt)


    small_hadlock_error = np.array(small_hadlock)-np.array(small_gt)
    large_hadlock_error = np.array(large_hadlock)-np.array(large_gt)
    normal_hadlock_error = np.array(normal_hadlock)-np.array(normal_gt)

    t_small_error = np.array(t_small_example)-np.array(small_gt)
    t_large_error = np.array(t_large_example)-np.array(large_gt)
    t_normal_error = np.array(t_normal_example)-np.array(normal_gt)

    # Compute metrics
    mae = compute_mae(pred_mem, gt_mem)
    rmse = compute_rmse(pred_mem, gt_mem)
    mape = compute_mape(pred_mem, gt_mem)
    mae_sd = compute_sd(np.abs(errors))
    rmse_sd = compute_sd(errors)
    mape_sd = compute_sd(np.abs(errors / gt_mem) * 100)
    percentage_within_200 = calculate_percentage_within_threshold(errors, 200)
    percentage_within_100 = calculate_percentage_within_threshold(errors, 100)

    t_mae = compute_mae(t_pred_mem, gt_mem)
    t_rmse = compute_rmse(t_pred_mem, gt_mem)
    t_mape = compute_mape(t_pred_mem, gt_mem)
    t_mae_sd = compute_sd(np.abs(t_errors))
    t_rmse_sd = compute_sd(t_errors)
    t_mape_sd = compute_sd(np.abs(t_errors / gt_mem) * 100)
    t_percentage_within_200 = calculate_percentage_within_threshold(t_errors, 200)
    t_percentage_within_100 = calculate_percentage_within_threshold(t_errors, 100)

    # Compute metrics for Hadlock
    hadlock_errors = hadlocks_mem - gt_mem
    hadlock_mae = compute_mae(hadlocks_mem, gt_mem)
    hadlock_rmse = compute_rmse(hadlocks_mem, gt_mem)
    hadlock_mape = compute_mape(hadlocks_mem, gt_mem)
    hadlock_mae_sd = compute_sd(np.abs(hadlock_errors))
    hadlock_rmse_sd = compute_sd(hadlock_errors)
    hadlock_mape_sd = compute_sd(np.abs(hadlock_errors / gt_mem) * 100)
    hadlock_percentage_within_200 = calculate_percentage_within_threshold(hadlock_errors, 200)
    hadlock_percentage_within_100 = calculate_percentage_within_threshold(hadlock_errors, 100

    def save_metrics_to_txt(filename, metrics):
        with open(filename, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    def save_experiment_results(save_dir):
        metrics = {
            "Weight MAE (x5000)": f"{mae:.4f} ± {mae_sd:.4f}",
            "Weight RMSE (x5000)": f"{rmse:.4f} ± {rmse_sd:.4f}",
            "Weight MAPE (x5000)": f"{mape:.4f}% ± {mape_sd:.4f}%",
            "Percentage of errors within 200": f"{percentage_within_200:.2f}%",
            "Percentage of errors within 100": f"{percentage_within_100:.2f}%",
            
            "Teacher Weight MAE (x5000)": f"{t_mae:.4f} ± {t_mae_sd:.4f}",
            "Teacher Weight RMSE (x5000)": f"{t_rmse:.4f} ± {t_rmse_sd:.4f}",
            "Teacher Weight MAPE (x5000)": f"{t_mape:.4f}% ± {t_mape_sd:.4f}%",
            "Teacher Percentage of errors within 200": f"{t_percentage_within_200:.2f}%",
            "Teacher Percentage of errors within 100": f"{t_percentage_within_100:.2f}%",
            
            "Hadlock MAE (x5000)": f"{hadlock_mae:.4f} ± {hadlock_mae_sd:.4f}",
            "Hadlock RMSE (x5000)": f"{hadlock_rmse:.4f} ± {hadlock_rmse_sd:.4f}",
            "Hadlock MAPE (x5000)": f"{hadlock_mape:.4f}% ± {hadlock_mape_sd:.4f}%",
            "Hadlock errors within 200": f"{hadlock_percentage_within_200:.2f}%",
            "Hadlock errors within 100": f"{hadlock_percentage_within_100:.2f}%",
        }
        
        if len(small_gt) != 0:
            metrics.update({
                "student small example error":  f"{compute_mae(s_small_example, small_gt):.4f} ± {compute_sd(s_small_error):.4f}",
                "teacher small example error":  f"{compute_mae(t_small_example, small_gt):.4f} ± {compute_sd(t_small_error):.4f}",
                "small hadlock error":  f"{compute_mae(small_hadlock, small_gt):.4f} ± {compute_sd(small_hadlock_error):.4f}",
                "small count": len(small_gt),
            })
        
        if len(large_gt) != 0:
            metrics.update({
                "student large example error":  f"{compute_mae(s_large_example, large_gt):.4f} ± {compute_sd(s_large_error):.4f}",
                "teacher large example error":  f"{compute_mae(t_large_example, large_gt):.4f} ± {compute_sd(t_large_error):.4f}",
                "large hadlock error":  f"{compute_mae(large_hadlock, large_gt):.4f} ± {compute_sd(large_hadlock_error):.4f}",
                "large count": len(large_gt),
            })
        
        if len(normal_gt) != 0:
            metrics.update({
                "student normal example error":  f"{compute_mae(s_normal_example, normal_gt):.4f} ± {compute_sd(s_normal_error):.4f}",
                "teacher normal example error":  f"{compute_mae(t_normal_example, normal_gt):.4f} ± {compute_sd(t_normal_error):.4f}",
                "normal hadlock error":  f"{compute_mae(normal_hadlock, normal_gt):.4f} ± {compute_sd(normal_hadlock_error):.4f}",
                "normal count": len(normal_gt),
            })
            
        save_metrics_to_txt(os.path.join(save_dir, "experiment_results.txt"), metrics)

    save_experiment_results(save_dir)
    save_predictions_to_json(pred_mem, t_pred_mem, save_dir)

    return t_pred_mem, gt_mem, hadlocks_mem, mae

def plot_predictions_vs_labels(predictions, labels, hadlocks, save_dir=None):
    plt.figure(figsize=(8, 8))
    plt.scatter(labels, predictions, alpha=0.5, label="Predictions")
    plt.scatter(labels, hadlocks, alpha=0.5, label="Hadlock", marker='^', color='green')
    plt.plot([min(labels), max(labels)], [min(labels), max(labels)], 'r--', label="y=x")
    plt.xlabel('True Labels')
    plt.ylabel('Values')
    plt.title('Predictions and Hadlock vs True Labels')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'predictions_and_hadlock_vs_labels.png'))

def plot_error_distribution(predictions, labels, hadlocks, save_dir=None):
    prediction_errors = predictions - labels
    hadlock_errors = hadlocks - labels

    plt.figure(figsize=(8, 6))
    plt.hist(prediction_errors, bins=50, alpha=0.7, color='blue', label="Prediction Error")
    plt.hist(hadlock_errors, bins=50, alpha=0.7, color='green', label="Hadlock Error")
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction and Hadlock Errors')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'error_distribution_with_hadlock.png'))

def main():
    root = r'/dataset'
    batch_size = 1
    body_parts = "abdomen"
    check = r'your check path'
    save_dir = '/'.join(check.split('/')[:-2])
    print(check)

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeanTeacher()
    model = model.to(device)
    target_shape = [160, 128, 96]

    checkpoint = torch.load(check, map_location=device)
    model.load_state_dict(checkpoint)

    save_path = os.path.join(save_dir,'results','test')
    os.makedirs(save_path, exist_ok=True)
    test_dataset = get_test_data(root, body_parts, target_shape)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    predictions, labels, hadlocks, mae = test(model, test_loader, device, save_path)
    plot_predictions_vs_labels(predictions, labels, hadlocks, save_dir=save_path)
    plot_error_distribution(predictions, labels, hadlocks, save_dir=save_path)
    plot_bland_altman(labels, predictions, save_path, 'Bland-Altman Plot of Predicted vs Actual Birth Weight')
    plot_consistency(labels, predictions, save_path, 'Consistency Plot of Predicted vs Actual Birth Weight')

    save_path = os.path.join(save_dir,'results','val')
    os.makedirs(save_path, exist_ok=True)
    test_dataset = get_validation_data(root, body_parts, target_shape)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    predictions, labels, hadlocks, mae = test(model, test_loader, device, save_path)
    plot_predictions_vs_labels(predictions, labels, hadlocks, save_dir=save_path)
    plot_error_distribution(predictions, labels, hadlocks, save_dir=save_path)
    plot_bland_altman(labels, predictions, save_path, 'Bland-Altman Plot of Predicted vs Actual Birth Weight')
    plot_consistency(labels, predictions, save_path, 'Consistency Plot of Predicted vs Actual Birth Weight')


    save_path = os.path.join(save_dir,'results','train')
    os.makedirs(save_path, exist_ok=True)
    test_dataset = get_training_data(root, body_parts, target_shape, is_aug=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    predictions, labels, hadlocks, mae = test(model, test_loader, device, save_path)
    plot_predictions_vs_labels(predictions, labels, hadlocks, save_dir=save_path)
    plot_error_distribution(predictions, labels, hadlocks, save_dir=save_path)
    plot_bland_altman(labels, predictions, save_path, 'Bland-Altman Plot of Predicted vs Actual Birth Weight')
    plot_consistency(labels, predictions, save_path, 'Consistency Plot of Predicted vs Actual Birth Weight')


if __name__ == "__main__":
    main()
