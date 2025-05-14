from torch.utils.data import DataLoader, Dataset
import torch


class TimeSeriesDataset(Dataset):
    def __init__(self, X_shuffled, y_shuffled, name_shuffled, open_shuffled):
        self.dataX = torch.tensor(X_shuffled, dtype=torch.float32)
        self.dataY = torch.tensor(y_shuffled, dtype=torch.float32)
        self.names = name_shuffled
        self.open = torch.tensor(open_shuffled, dtype=torch.float32)
        
    def __len__(self):
        return self.dataX.size(0)

    def __getitem__(self, idx):
        sequence = self.dataX[idx, :]
        target = self.dataY[idx]
        name = self.names[idx]
        open = self.open[idx]
        return sequence, target, name, open