import torch.nn.functional as F
import torch.nn as nn
import torch



class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, lstm_outputs):
        # lstm_outputs: [batch_size, seq_len, hidden_dim]
        attention_scores = self.attention_weights(lstm_outputs).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        weighted_output = torch.sum(lstm_outputs * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]
        return weighted_output, attention_weights
    
class LSTMDoubleAttentionModel(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False):
        super(LSTMDoubleAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        # Embedding layer (optional: use if input is categorical or needs projection)
        self.embedding = nn.Linear(input_dim, embed_dim)  # Replace with nn.Embedding if needed
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                                    batch_first=True, bidirectional=bidirectional)
        
        # Attention
        self.attention = AttentionMechanism(hidden_dim * self.directions)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(hidden_dim * self.directions, hidden_dim, num_layers,
                                    batch_first=True, bidirectional=bidirectional)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * self.directions, output_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Encoder LSTM
        encoder_out, _ = self.encoder_lstm(embedded)  # [batch_size, seq_len, hidden_dim * directions]
        
        # Attention
        attention_out, attention_weights = self.attention(encoder_out)  # [batch_size, hidden_dim * directions]
        
        # Prepare Decoder Input (sequence length 1)
        decoder_input = attention_out.unsqueeze(1)  # [batch_size, 1, hidden_dim * directions]
        
        # Decoder LSTM
        decoder_out, _ = self.decoder_lstm(decoder_input)  # [batch_size, 1, hidden_dim * directions]
        
        # Final Prediction
        output = self.fc(decoder_out.squeeze(1))  # [batch_size, output_dim]
        
        return output, attention_weights