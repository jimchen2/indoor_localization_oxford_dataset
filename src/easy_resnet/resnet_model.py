import torch
import torch.nn as nn

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn_residual = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.bn_residual(self.residual(x))
        out += residual
        out = self.relu(out)
        return out

class IMUResNetModel(nn.Module):
    def __init__(self, input_size, channels, output_size, dropout_rate=0.5):
        super(IMUResNetModel, self).__init__()
        self.device = get_device()
        
        self.normalize = nn.BatchNorm1d(input_size)
        
        self.conv1 = nn.Conv1d(input_size, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            stride = 2 if i > 0 else 1
            self.layers.append(self._make_layer(channels[i], channels[i+1], 2, stride))
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.to(self.device)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.to(self.device)
        x = x.permute(0, 2, 1)
        
        x = self.normalize(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

    def to(self, device):
        self.device = device
        return super(IMUResNetModel, self).to(device)