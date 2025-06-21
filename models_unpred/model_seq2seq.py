import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class MultiStepSiameseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_patterns, n_steps=1, num_heads=4):
        super().__init__()
        self.num_patterns = num_patterns
        self.hidden_size = hidden_size
        self.n_steps = n_steps

        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            num_layers=2
        )

        self.pattern_context = nn.Parameter(
            torch.randn(num_patterns, hidden_size))

        self.decoder_cell = nn.LSTMCell(
            input_size + hidden_size, 
            hidden_size
        )
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.2
        )
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.shared_norm = nn.LayerNorm(hidden_size)
        self.attention_weights = nn.Parameter(torch.ones(num_patterns))

    def _process_pattern(self, pattern_input):
        enc_out, (h_n, c_n) = self.encoder(pattern_input)
        last_out = enc_out[:, -1, :]
        return self.shared_norm(last_out), (h_n, c_n)

    def forward(self, inputs, targets=None, teacher_forcing_ratio=0.5):
        pattern_outputs = []
        pattern_states = []
        for x in inputs:
            out, state = self._process_pattern(x)
            pattern_outputs.append(out)
            pattern_states.append(state)

        stacked = torch.stack(pattern_outputs, dim=1)

        attn_output, attn_weights = self.attention(stacked, stacked, stacked)
        weights = F.softmax(self.attention_weights, dim=0)
        context = torch.sum(attn_output * weights[None, :, None], dim=1)

        batch_size = inputs[0].size(0)
        device = inputs[0].device

        last_value = inputs[0][:, -1, -1].unsqueeze(1)  
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        cell = torch.zeros(batch_size, self.hidden_size).to(device)

        predictions = []
        individual_step_preds = []  

        for step in range(self.n_steps):
            decoder_input = torch.cat([last_value, context], dim=1)

            hidden, cell = self.decoder_cell(decoder_input, (hidden, cell))

            combined = torch.cat([hidden, context], dim=1)
            output = self.output_fc(combined)

            pattern_preds = []
            for i in range(len(inputs)):  
                pattern_ctx = attn_output[:, i, :]
                pattern_combined = torch.cat([hidden, pattern_ctx], dim=1)
                pattern_pred = self.output_fc(pattern_combined)
                pattern_preds.append(pattern_pred)

            individual_step_preds.append(torch.stack(pattern_preds, dim=1))
            predictions.append(output)

            use_teacher_forcing = (targets is not None and
                                  torch.rand(1).item() < teacher_forcing_ratio)
            if use_teacher_forcing:
                last_value = targets[:, step].unsqueeze(1)
            else:
                last_value = output.detach()

        return (
            torch.cat(predictions, dim=1),  
            torch.stack(individual_step_preds, dim=2),  
            attn_weights  
        )

    def simple_multi_step_predict(self, initial_series, patterns, steps):
        """Forsed предсказание MIMO"""
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_series = initial_series.copy()
        predictions = []
        pattern_tuples = [tuple(p) for p in patterns]
        step = 0

        while step < steps:
            steps_ = self.n_steps - 1
            last_inputs, predict_index = self.generate_last_inputs(current_series, pattern_tuples, steps_)
            tensor_inputs = [
                torch.tensor(arr, dtype=torch.float32).to(device).view(1, len(arr), 1)
                for arr in last_inputs
            ]

            with torch.no_grad():
                pred, _, _ = self(tensor_inputs) 

            chunk_preds = pred[0].tolist()
            chunk_size = len(chunk_preds)
            steps_to_predict = min(chunk_size, steps - step)

            for i in range(steps_to_predict):
                next_val = chunk_preds[i]
                predictions.append(next_val)

                target_idx = predict_index + i
                if target_idx < len(current_series):
                    current_series[target_idx] = next_val
                else:
                    if target_idx >= len(current_series):
                        padding = [0.0] * (target_idx - len(current_series) + 1)
                        current_series = np.append(current_series, padding)
                    current_series[target_idx] = next_val

            step += steps_to_predict

        return np.array(predictions[:steps])

    def predict_multi_step_with_analysis(
        self,
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
        """Monte Carlo предсказание с кластеризацией """
        self.train() 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pattern_tuples = [tuple(p) for p in patterns]
        mc_predictions = np.zeros((n_mc, steps))
        degenerate_flags = np.zeros((n_mc, steps), dtype=bool)

        for mc_run in range(n_mc):
            current_series = initial_series.copy()
            full_preds = []
            step = 0

            while step < steps:
                steps_ = self.n_steps - 1
                last_inputs, predict_index = self.generate_last_inputs(current_series, pattern_tuples, steps_)
                tensor_inputs = [
                    torch.tensor(arr, dtype=torch.float32).to(device).view(1, len(arr), 1)
                    for arr in last_inputs
                ]

                with torch.no_grad():
                    pred, _, _ = self(tensor_inputs) 

                chunk_preds = pred[0].tolist()
                chunk_size = len(chunk_preds)
                steps_to_predict = min(chunk_size, steps - step)

                for i in range(steps_to_predict):
                    next_val = chunk_preds[i]
                    full_preds.append(next_val)

                    target_idx = predict_index + i
                    if target_idx < len(current_series):
                        current_series[target_idx] = next_val
                    else:
                        if target_idx >= len(current_series):
                            padding = [0.0] * (target_idx - len(current_series) + 1)
                            current_series = np.append(current_series, padding)
                        current_series[target_idx] = next_val

                step += steps_to_predict

            mc_predictions[mc_run, :] = np.array(full_preds[:steps])

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

    def robust_multi_step_predict(
        self,
        initial_series,
        patterns,
        steps,
        variance_threshold=0.01,
        replacement_strategy="linear",
        window_size=15
    ):
        """Функция для определения непрогнозируемых точек по разбросу"""
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_series = initial_series.copy()
        raw_predictions = []
        corrected_predictions = []
        uncertainty_flags = []
        all_variances = []
        last_reliable_idx = len(initial_series) - 1
        last_reliable_value = initial_series[-1]
        reliable_history = []  
        pattern_tuples = [tuple(p) for p in patterns]
        step = 0

        while step < steps:
            steps_ = self.n_steps-1
            last_inputs, predict_index = self.generate_last_inputs(current_series, pattern_tuples, steps_)
            tensor_inputs = [
                torch.tensor(arr, dtype=torch.float32).to(device).view(1, len(arr), 1)
                for arr in last_inputs
            ]

            with torch.no_grad():
                pred, individual_preds, attn_weights = self(tensor_inputs)

                weights = F.softmax(self.attention_weights, dim=0)
                mean_pred = torch.sum(individual_preds * weights, dim=1, keepdim=True)
                weighted_variance = torch.sum(
                    weights * (individual_preds - mean_pred)**2,
                    dim=1
                ).mean().item()

                is_uncertain = weighted_variance >= variance_threshold

            chunk_preds = pred[0].tolist()
            chunk_size = len(chunk_preds)
            steps_to_predict = min(chunk_size, steps - step)

            for i in range(steps_to_predict):
                raw_val = chunk_preds[i]
                raw_predictions.append(raw_val)
                uncertainty_flags.append(is_uncertain)
                all_variances.append(weighted_variance)

                if is_uncertain:
                    if replacement_strategy == "last":
                        replacement = last_reliable_value
                    elif replacement_strategy == "mean" and reliable_history:
                        replacement = np.mean(reliable_history[-window_size:])
                    elif replacement_strategy == "linear" and len(reliable_history) >= 2:
                        x = np.arange(len(reliable_history[-2:]))
                        y = reliable_history[-2:]
                        coeffs = np.polyfit(x, y, 1)
                        replacement = np.polyval(coeffs, 2)
                    else: 
                        replacement = last_reliable_value
                else:
                    replacement = raw_val
                    last_reliable_value = raw_val
                    last_reliable_idx = len(current_series) + i
                    reliable_history.append(raw_val)

                corrected_predictions.append(replacement)

                target_idx = predict_index + i
                if target_idx < len(current_series):
                  current_series[target_idx] = replacement
                else:
                    if target_idx >= len(current_series):
                      padding = [0.0] * (target_idx - len(current_series) + 1)
                      current_series = np.append(current_series, padding)
                    current_series[target_idx] = replacement

            step += steps_to_predict

        return {
            "raw_predictions": np.array(raw_predictions),
            "corrected_predictions": np.array(corrected_predictions),
            "unpredictable_flags": np.array(uncertainty_flags),
            "variances": np.array(all_variances)
        }
    @staticmethod
    def generate_last_inputs(time_series, patterns, n_steps):
        max_history = max(sum(p) for p in patterns)
        n = len(time_series)
        predict_index = n

        last_inputs = []
        for pattern in patterns:
            history_steps = sum(pattern[:-1])
            start_index = n-history_steps-1

            indices = []
            current_idx = start_index
            for step in pattern[:-n_steps-1]:
                indices.append(current_idx)
                current_idx += step
            indices.append(current_idx)
            input_vector = time_series[indices]
            last_inputs.append(input_vector)

        return last_inputs, predict_index
