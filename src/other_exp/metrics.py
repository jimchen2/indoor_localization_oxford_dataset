import numpy as np
from scipy.spatial.transform import Rotation

def calculate_ate(pred_trajectory, gt_trajectory):
    error = np.linalg.norm(pred_trajectory - gt_trajectory, axis=1)
    return np.mean(error), np.max(error)

def calculate_rte(pred_trajectory, gt_trajectory, time_step=0.01, interval=60):
    pred_trajectory = np.asarray(pred_trajectory)
    gt_trajectory = np.asarray(gt_trajectory)
    
    assert pred_trajectory.shape == gt_trajectory.shape, "Trajectories must have the same shape"
    
    interval_steps = int(interval / time_step)
    sequence_duration = len(pred_trajectory) * time_step
    if sequence_duration < interval:
        last_error = np.linalg.norm(pred_trajectory[-1] - gt_trajectory[-1])
        scaling_factor = interval / sequence_duration
        return last_error * scaling_factor, last_error * scaling_factor
    
    errors = []
    for i in range(0, len(pred_trajectory) - interval_steps + 1, interval_steps):
        end_idx = i + interval_steps
        pred_interval = pred_trajectory[i:end_idx]
        gt_interval = gt_trajectory[i:end_idx]
        mse = np.mean(np.sum((pred_interval - gt_interval)**2, axis=1))
        rmse = np.sqrt(mse)
        errors.append(rmse)
    
    return np.mean(errors), np.max(errors)

def calculate_mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def calculate_mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def calculate_rmse(predictions, targets):
    return np.sqrt(calculate_mse(predictions, targets))

def calculate_metrics(predictions, targets):
    ate_mean, ate_max = calculate_ate(predictions, targets)
    rte_mean, rte_max = calculate_rte(predictions, targets)
    mse = calculate_mse(predictions, targets)
    mae = calculate_mae(predictions, targets)
    rmse = calculate_rmse(predictions, targets)

    metrics = {
        'ATE_mean': ate_mean,
        'ATE_max': ate_max,
        'RTE_mean': rte_mean,
        'RTE_max': rte_max,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    }

    return metrics  