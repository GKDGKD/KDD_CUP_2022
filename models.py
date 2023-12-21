import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, 
                 hidden_size, 
                 output_size,
                 num_layers=1,
                 activation = 'tanh',
                 dropout=0,
                 device='cpu',
                 batch_first=True
                 ):
        super(RNN, self).__init__()
        assert activation in ['tanh', 'relu']
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.device      = device
        self.dropout     = dropout
        self.activation  = activation
        self.batch_first = batch_first
        self.model = nn.Sequential(
            nn.RNN(input_size=self.input_size, 
                   hidden_size  = self.hidden_size,
                   num_layers   = self.num_layers,
                   nonlinearity = self.activation,
                   dropout      = self.dropout,
                   batch_first  = self.batch_first),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        x = x.to(self.device)
        self.model.to(self.device)
        
        # x: (batch_size, sequence_length, input_size)
        out, _ = self.model[0](x)
        # out: (batch_size, sequence_length, hidden_size)
        out = self.model[1](out[:, -1, :])
        # out: (batch_size, output_size)
        return out.to(self.device)
    
class LSTM(nn.Module):
    def __init__(self, input_size, 
                 hidden_size, 
                 output_size,
                 num_layers=1,
                 activation = 'tanh',
                 dropout=0,
                 device='cpu',
                 batch_first=True
                 ):
        super(LSTM, self).__init__()
        assert activation in ['tanh', 'relu']
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.device      = device
        self.dropout     = dropout
        self.activation  = activation
        self.batch_first = batch_first
        self.model = nn.Sequential(
            nn.LSTM(input_size=self.input_size, 
                   hidden_size  = self.hidden_size,
                   num_layers   = self.num_layers,
                   nonlinearity = self.activation,
                   dropout      = self.dropout,
                   batch_first  = self.batch_first),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        x = x.to(self.device)
        self.model.to(self.device)
        
        # x: (batch_size, sequence_length, input_size)
        out, _ = self.model[0](x)
        # out: (batch_size, sequence_length, hidden_size)
        out = self.model[1](out[:, -1, :])
        # out: (batch_size, output_size)
        return out.to(self.device)