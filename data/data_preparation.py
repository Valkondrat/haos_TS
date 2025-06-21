import numpy as np
from itertools import product
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import math
import torch.nn.functional as F


def generate_generalized_z_vectors(time_series, pattern):
    """Функция генерирует обобщённые z-вектора на основе заданного паттерна"""
    pattern = np.array(pattern)
    max_index_offset = pattern.sum()
    n_vectors = len(time_series) - max_index_offset

    if n_vectors <= 0:
        raise ValueError("Pattern is too long for the given time series.")
    cumulative_offsets = np.concatenate(([0], np.cumsum(pattern)))
    z_vectors = np.array([
        time_series[i + cumulative_offsets] for i in range(n_vectors)
    ])

    return z_vectors

def grid_search_patterns_fixed(time_series, max_lag, embedding_dim):
    """"Функция формирует паттерны на основе max_lag и embedding_dim и генерирует тройку (паттерн, прото-обучающая выборкаб прото-у) (пункт 4 в пайплайне)"""
    possible_patterns = list(product(range(1, max_lag + 1), repeat=embedding_dim - 1))
    results = []

    for pattern in possible_patterns:
        max_pattern = np.array(possible_patterns[-1]).sum()
        z_vectors = generate_generalized_z_vectors(time_series, pattern)
        z_vectors = z_vectors[z_vectors.shape[0]-(len(time_series) - max_pattern):]
        if z_vectors.size > 0:
            results.append((pattern, torch.tensor(z_vectors[:, :-1], dtype=torch.float32).unsqueeze(-1), torch.tensor(z_vectors[:, -1], dtype=torch.float32)))

    return results

def get_batches_from_dataloaders(dataloaders, n_batches):
    """"Функция формирует батчи на основе прото выборки (пункт 5 в пайплайне)"""
    all_batches = []
    for i, dl in enumerate(dataloaders):
        batches = []
        loader_iter = iter(dl)
        for n in range(n_batches):
            try:
                batch = next(loader_iter)
                batches.append(batch)
            except StopIteration:
                print(f"DataLoader {i + 1} exhausted before reaching {n_batches} batches.")
                break
        all_batches.append(batches)
    return all_batches

def make_dataset(train, test, seed_worker=None, batch_size = 128, embedding_dim = 5, max_lag = 3):
  """Функция для создания итогового датасета для обучения для одношагового прогноза (логику см.Глава 2.2)"""
  train_results = grid_search_patterns_fixed(train, max_lag, embedding_dim)
  test_results = grid_search_patterns_fixed(test, max_lag, embedding_dim)
  dataloaders_train = []

  for pattern, X_train, y_train in train_results:
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, worker_init_fn=seed_worker, shuffle=False)
    dataloaders_train.append(dataloader)

  dataloaders_test = []
  for pattern, X_test, y_test in test_results:
    dataset_test = TensorDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, worker_init_fn=seed_worker, shuffle=False)
    dataloaders_test.append(dataloader)

  n_batches = len(dataset) // batch_size
  train_batches = get_batches_from_dataloaders(dataloaders_train, n_batches)

  n_batches = len(dataset_test) // batch_size
  val_batches = get_batches_from_dataloaders(dataloaders_test, n_batches)

  return train_results, test_results, train_batches, val_batches

def generate_last_inputs(time_series, patterns):
    """Функция для создания выборки для рекурсивного прогнозирования. Создает такую же структуру, какую модель видела на этапе обучения"""
    max_history = max(sum(p) for p in patterns)
    n = len(time_series)
    predict_index = n

    last_inputs = []
    for pattern in patterns:
        history_steps = sum(pattern[:-1])

        start_index = n - max_history + (max_history - history_steps - pattern[-1])

        indices = []
        current_idx = start_index
        for step in pattern[:-1]:
            indices.append(current_idx)
            current_idx += step

        indices.append(current_idx)

        input_vector = time_series[indices]
        last_inputs.append(input_vector)

    return last_inputs, predict_index

def grid_search_patterns_mult(time_series, max_lag, embedding_dim, n):
    """"Функция формирует паттерны на основе max_lag и embedding_dim и генерирует тройку (паттерн, прото-обучающая выборкаб прото-у) (пункт 4 в пайплайне) 
    для Seq2Seq модели"""
    base_length = embedding_dim - n
    base_patterns = list(product(range(1, max_lag + 1), repeat=base_length))
    possible_patterns = [base + (1,)*n for base in base_patterns]
    max_pattern_val = max_lag * base_length + n if base_length > 0 else n

    results = []
    for pattern in possible_patterns:
        z_vectors = generate_generalized_z_vectors(time_series, pattern)

        if z_vectors.size == 0:
            continue

        n_keep = len(time_series) - max_pattern_val
        if n_keep <= 0:
            continue
        z_vectors = z_vectors[-n_keep:]

        inputs = z_vectors[:, :-(n+1)]
        target = z_vectors[:, -(n+1):]

        results.append((
            pattern,
            torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        ))

    return results

def make_dataset_mult(train, test,seed_worker = None, batch_size = 128, embedding_dim = 5, max_lag = 3, n = 4):
  """Функция для создания итогового датасета для обучения для Seq2Seq (логику см.Глава 2.2)"""
  train_results = grid_search_patterns_mult(train, max_lag, embedding_dim, n)
  test_results = grid_search_patterns_mult(test, max_lag, embedding_dim,n)
  dataloaders = []
  for pattern, X_train, y_train in train_results:
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size,worker_init_fn=seed_worker, shuffle=False)
    dataloaders.append(dataloader)
  dataloaders_test = []
  for pattern, X_test, y_test in test_results:
    dataset_test = TensorDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size,worker_init_fn=seed_worker, shuffle=False)
    dataloaders_test.append(dataloader)

  n_batches = len(dataset) // batch_size
  train_batches = get_batches_from_dataloaders(dataloaders, n_batches)
  n_batches = len(dataset_test) // batch_size
  val_batches = get_batches_from_dataloaders(dataloaders_test, n_batches)
  return train_results, test_results, train_batches, val_batches
