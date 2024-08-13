import torch
from torch import nn
from tcn import TemporalConvNet

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IMUTCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, output_size, dropout_rate=0.2):
        super(IMUTCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout_rate)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.device = get_device()
        self.to(self.device)

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        # TCN expects input shape (batch, channels, seq_len)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # Use the last output for prediction
        return self.linear(y[:, :, -1])

    def to(self, device):
        self.device = device
        return super(IMUTCNModel, self).to(device)