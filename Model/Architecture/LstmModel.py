import torch.nn.functional as F
import torch.nn as nn
import torch


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]  # Take the output at the last time step
        return self.fc(last_output)