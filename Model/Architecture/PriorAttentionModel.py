import torch.nn.functional as F
import torch.nn as nn
import torch

#TODO test and utilize
class AttentionMechanism(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionMechanism, self).__init__()
        self.attention_weights = nn.Linear(embed_dim, 1, bias=False)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        attention_scores = self.attention_weights(x).squeeze(-1) # [batch_size, seq_len]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        weighted_output = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, embed_dim]
        return weighted_output # [batch_size, seq_len]
    
    
class LSTMDoubleAttentionModel(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False):
        super(LSTMDoubleAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        # x shape: [batch_size, seq_len, input_dim]
        self.embedding = nn.Linear(input_dim, embed_dim)  # [batch_size, seq_len, embed_dim]
        
        self.attention = AttentionMechanism(embed_dim) # [batch_size, embed_dim]

        
        # # Decoder LSTM
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                                    batch_first=True, bidirectional=bidirectional) # [batch_size, hidden_dim * directions]
        
        # Attention
    
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * self.directions, output_dim)

    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        x = self.attention(x)  # # [batch_size, embed_dim]
        # # Encoder LSTM
        x, _ = self.encoder_lstm(x) # [batch_size, hidden_dim * directions]

        
        # # Attention
        
        # # Prepare Decoder Input (sequence length 1)
        # x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim * directions]
        
        # # Decoder LSTM
        # x, _ = self.decoder_lstm(x)  # [batch_size, 1, hidden_dim * directions]
        
        # # Final Prediction
        output = self.fc(x)  # [batch_size, output_dim]
        
        return output