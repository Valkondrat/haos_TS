import torch

def train_one_step(model, criterion, optimizer, train_batches, val_batches, num_epochs = 10):
    """Функция для обучения одношаговой модели"""
    for epoch in range(num_epochs):
      model.train()
      for i in range(len(train_batches[0])):
        cur_int = [item[i] for item in train_batches]
        X_bathes = [item[0] for item in cur_int]
        y_batches = [item[1] for item in cur_int]
        outputs, individual_preds, attention_w = model(X_bathes, return_individual=True)
        loss = criterion(outputs.squeeze(-1), y_batches[-1])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model

def train_mult_step(model, criterion, optimizer,train_batches,val_batches, teacher_forcing_decay = 0.95, teacher_forcing_ratio = 0.5, num_epochs = 10):
    """Функция для обучения Seq2Seq модели"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
      model.train()
      epoch_loss = 0.0
      current_tf_ratio = teacher_forcing_ratio * (teacher_forcing_decay ** epoch)

      for i in range(len(train_batches[0])):
          cur_int = [item[i] for item in train_batches]
          X_batches = [item[0].to(device) for item in cur_int]
          y_target = cur_int[-1][1].to(device)

          outputs,_, _ = model(X_batches, targets=y_target.squeeze(-1),
                          teacher_forcing_ratio=current_tf_ratio)

          loss = criterion(outputs.unsqueeze(-1), y_target)

          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
          optimizer.step()

          epoch_loss += loss.item()

    return model
