from data.data_preparation import generate_last_inputs

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


def predict_multi_step(model, initial_series, patterns, steps):
    """Forsed предсказания"""
    model.eval()
    current_series = initial_series.copy()
    predictions = []
    pattern_tuples = [tuple(p) for p in patterns]

    for step in range(steps):
        last_inputs, predict_index = generate_last_inputs(current_series, pattern_tuples)

        tensor_inputs = [
            torch.tensor(arr, dtype=torch.float32).view(1, len(arr), 1)
            for arr in last_inputs
        ]

        with torch.no_grad():
            pred = model(tensor_inputs)
        next_val = pred.squeeze().item()

        if predict_index < len(current_series):
            current_series[predict_index] = next_val
        else:
            current_series = np.append(current_series, next_val)

        predictions.append(next_val)

    return np.array(predictions)

def robust_multi_step_predict(model, initial_series, patterns, steps, variance_threshold=0.01,
                              replacement_strategy="linear", window_size=15):
    """Функция для определения непрогнозируемых точек по разбросу"""
    model.eval()
    current_series = initial_series.copy()
    predictions = []
    reliable_predictions = [] 
    uncertainty_flags = []
    all_variances = []
    last_reliable_idx = len(initial_series) - 1
    last_reliable_value = initial_series[-1]
    reliable_history = []  

    pattern_tuples = [tuple(p) for p in patterns]

    for step in range(steps):
        last_inputs, predict_index = generate_last_inputs(current_series, pattern_tuples)

        tensor_inputs = [
            torch.tensor(arr, dtype=torch.float32).view(1, len(arr), 1)
            for arr in last_inputs
        ]

        with torch.no_grad():
            pred, individual_preds, _ = model(tensor_inputs, return_individual=True)

            weights = F.softmax(model.attention_weights, dim=0).detach()
            mean_pred = torch.sum(individual_preds * weights, dim=1, keepdim=True)
            weighted_variance = torch.sum(
                weights * (individual_preds - mean_pred)**2,
                dim=1
            ).mean().item()

            is_uncertain = weighted_variance >= variance_threshold

        next_val = pred.squeeze().item()
        predictions.append(next_val)
        uncertainty_flags.append(is_uncertain)
        all_variances.append(weighted_variance)

        if is_uncertain:
            if replacement_strategy == "last":
                replacement = last_reliable_value

            elif replacement_strategy == "mean" and reliable_history:
                replacement = np.mean(reliable_history[-window_size:])
            else:  
                replacement = last_reliable_value

            reliable_predictions.append(replacement)
        else:
            replacement = next_val
            reliable_predictions.append(next_val)
            last_reliable_value = next_val
            last_reliable_idx = len(current_series)
            reliable_history.append(next_val)

        current_series = np.append(current_series, replacement)

    return {
        "raw_predictions": np.array(predictions),
        "corrected_predictions": np.array(reliable_predictions),
        "unpredictable_flags": np.array(uncertainty_flags),
        "variances": np.array(all_variances)
    }

def predict_and_analyze(
    model, 
    initial_series, 
    patterns, 
    steps, 
    n_mc=30,
    degenerate_threshold=1e-5,
    cluster_variance_threshold=0.01,
    min_cluster_ratio=0.6,
    min_valid_trajectories=5,
    degenerate_prop_threshold=0.5,
    cluster_ratio_threshold=0.6
):
    """Функция для формирования предсказания на основе Monte Carlo Dropout и кластеризации полученных траекторий  + отсечение дегенерировавших траекторий"""
    model.train()  
    pattern_tuples = [tuple(p) for p in patterns]
    mc_predictions = np.zeros((n_mc, steps))
    degenerate_flags = np.zeros((n_mc, steps), dtype=bool)
    
    for mc_run in range(n_mc):
        current_series = initial_series.copy()
        preds = []
        for step in range(steps):
            last_inputs, predict_index = generate_last_inputs(current_series, pattern_tuples)
            tensor_inputs = [torch.tensor(arr, dtype=torch.float32).view(1, len(arr), 1) 
                            for arr in last_inputs]
            with torch.no_grad():
                pred = model(tensor_inputs)
            next_val = pred.squeeze().item()
            preds.append(next_val)
            if predict_index < len(current_series):
                current_series[predict_index] = next_val
            else:
                current_series = np.append(current_series, next_val)
        mc_predictions[mc_run, :] = np.array(preds)
    
    for i in range(n_mc):
        for j in range(1, steps):
            window = mc_predictions[i, max(0, j-2):j+1]
            if len(window) >= 2:
                degenerate_flags[i, j] = np.var(window) < degenerate_threshold
    
    corrected = np.zeros(steps)
    cluster_assignments = -np.ones((steps, n_mc), dtype=int)
    unpredictability_flags = np.zeros(steps, dtype=bool)
    
    for j in range(steps):
        deg_prop = np.mean(degenerate_flags[:, j])
        valid_mask = ~degenerate_flags[:, j]
        valid_trajs = mc_predictions[valid_mask, j]
        n_valid = len(valid_trajs)
        
        if n_valid == 0:
            unpredictability_flags[j] = True
            corrected[j] = corrected[j-1] if j > 0 else initial_series[-1]
            continue
        elif deg_prop > degenerate_prop_threshold:
            unpredictability_flags[j] = True
        elif n_valid < min_valid_trajectories:
            unpredictability_flags[j] = True
        
        overall_var = np.var(valid_trajs)
        n_clusters = 2 if overall_var > cluster_variance_threshold else 1
        
        if n_clusters > 1 and n_valid > 1:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            cluster_labels = kmeans.fit_predict(valid_trajs.reshape(-1, 1))
            
            valid_indices = np.where(valid_mask)[0]
            for idx, label in zip(valid_indices, cluster_labels):
                cluster_assignments[j, idx] = label
            
            cluster_counts = np.bincount(cluster_labels.astype(int)) 
            main_cluster_ratio = np.max(cluster_counts) / n_valid
            
            if main_cluster_ratio < cluster_ratio_threshold:
                unpredictability_flags[j] = True
        else:
            cluster_labels = np.zeros(n_valid, dtype=int)  
            
        if not unpredictability_flags[j]:
            if n_valid > 0:
                cluster_counts = np.bincount(cluster_labels.astype(int))
                main_cluster = np.argmax(cluster_counts)
                cluster_values = valid_trajs[cluster_labels == main_cluster]
                corrected[j] = np.mean(cluster_values)
            else:
                corrected[j] = np.mean(valid_trajs)
        else:
            corrected[j] = corrected[j-1] if j > 0 else initial_series[-1]
    
    return {
        'corrected_predictions': corrected,
        'mc_predictions': mc_predictions,
        'cluster_assignments': cluster_assignments,
        'degenerate_flags': degenerate_flags,
        'unpredictable_flags': unpredictability_flags,
        'unpredictable_indices': np.where(unpredictability_flags)[0]
    }
