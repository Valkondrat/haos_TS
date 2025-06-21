import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseLSTMWithAttention(nn.Module):
    """Класс модели для одношагового прогнозирования"""
    def __init__(self, input_size, hidden_size, num_patterns, num_heads=4):
        super().__init__()
        self.num_patterns = num_patterns
        self.hidden_size = hidden_size

        self.shared_lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            dropout=0.2,
            num_layers=3
        )

        self.shared_norm = nn.LayerNorm(hidden_size)
        self.shared_fc = nn.Linear(hidden_size, 1)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.3
        )

        self.attention_weights = nn.Parameter(torch.ones(num_patterns))

    def forward(self, inputs, return_individual=False):
        branch_outputs = []
        for pattern_input in inputs:
            lstm_out, _ = self.shared_lstm(pattern_input)
            last_out = lstm_out[:, -1, :]
            normalized_out = self.shared_norm(last_out)
            branch_outputs.append(normalized_out)

        stacked = torch.stack(branch_outputs, dim=1)

        attn_output, attn_weights = self.attention(
            query=stacked,
            key=stacked,
            value=stacked
        )

        individual_preds = [self.shared_fc(out) for out in branch_outputs]
        individual_stacked = torch.stack(individual_preds, dim=1).squeeze(-1) 

        weights = F.softmax(self.attention_weights, dim=0)
        aggregated = torch.sum(attn_output * weights[None, :, None], dim=1)

        combined_pred = self.shared_fc(aggregated)

        if return_individual:
            return combined_pred, individual_stacked, attn_weights
        return combined_pred
