import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, names):
        self.dataX = torch.tensor(X, dtype=torch.float32)
        self.dataY = torch.tensor(y, dtype=torch.float32)
        self.names = names
        
    def __len__(self):
        return self.dataX.size(0)

    def __getitem__(self, idx):
        sequence = self.dataX[idx, :]
        target = self.dataY[idx]
        name = self.names[idx]
        return sequence, target, name

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        normalized = self.batch_norm(last_output)
        dropped = self.dropout(normalized)
        return self.fc(dropped)

def validate(model,criterion,val_loader):
    model.eval()
    model.to(device)
    val_loss = 0
    DA=0

    
def train_one_epoch(model, optimizer, scheduler, criterion, train_loader, clip_value=1.0):
    model.train()
    epoch_loss = 0
    
    for sequences, targets, name in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = model(sequences)
        mse = criterion(outputs.squeeze(), targets)
        dir_loss = directional_loss(outputs, targets, sequences)
        
        loss = mse + alpha * dir_loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    return epoch_loss

# Updated hyperparameters
hyperparameters = {
    'lookback_window': 14,  # Increased from 7
    'input_size': 7,
    'output_size': 1,
    'hidden_size': 128,     # Increased from 100
    'num_layers': 3,        # Reduced from 10
    'learning_rate': 0.001, # Increased from 0.0001
    'epochs': 100,          # Increased from 50
    'batch_size': 64,       # Increased from 32
    'dropout': 0.2
}

# Initialize model with updated architecture
model = LSTMModel(
    hyperparameters['input_size'],
    hyperparameters['hidden_size'],
    hyperparameters['num_layers'],
    hyperparameters['output_size'],
    hyperparameters['dropout']
)

# Initialize optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=hyperparameters['learning_rate'],
    weight_decay=0.01
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# Loss function with smooth L1 loss (Huber loss)
criterion = nn.SmoothL1Loss()

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training loop with early stopping
early_stopping = EarlyStopping(patience=10)

for epoch in range(hyperparameters['epochs']):
    epoch_loss = train_one_epoch(model, optimizer, scheduler, criterion, train_loader)
    val_loss, direction = validate(model, criterion, val_loader)
    
    print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, "
          f"Loss: {epoch_loss/len(train_loader):.4f}, "
          f"Validation Loss: {val_loss/len(val_loader):.4f}, "
          f"Directional Accuracy: {direction/len(nameAll):.4f}")
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break