import torch
import torch.nn as nn

def conv3(in_planes, out_planes, kernel_size, stride=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size // 2, bias=False)

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None, dropout_rate1=0.1, dropout_rate2=0.1):        
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv3(in_planes, out_planes, kernel_size, stride, dilation)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_rate1)
        self.conv2 = conv3(out_planes, out_planes, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.dropout2 = nn.Dropout(p=dropout_rate2)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.dropout2(out)
        return out

class FCOutputModule(nn.Module):
    def __init__(self, in_planes, num_outputs, fc_dim=1024, dropout_rate=0.5):
        super(FCOutputModule, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_planes, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_dim, num_outputs))

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y

class ResNet1D(nn.Module):
    def __init__(self, num_inputs, num_outputs, block_type, group_sizes, base_plane=64, output_block=None,
                 zero_init_residual=False, **kwargs):
        super(ResNet1D, self).__init__()
        self.base_plane = base_plane
        self.inplanes = self.base_plane

        input_dropout_rate = kwargs.get('input_dropout_rate', 0.1)
        self.input_block = nn.Sequential(
            nn.Conv1d(num_inputs, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.Dropout(p=input_dropout_rate),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.planes = [self.base_plane * (2 ** i) for i in range(len(group_sizes))]
        kernel_size = kwargs.get('kernel_size', 3)
        strides = [1] + [2] * (len(group_sizes) - 1)
        dilations = [1] * len(group_sizes)
        dropout_rates1 = kwargs.get('dropout_rates1', [0.1] * len(group_sizes))
        dropout_rates2 = kwargs.get('dropout_rates2', [0.1] * len(group_sizes))
        groups = [self._make_residual_group1d(block_type, self.planes[i], kernel_size, group_sizes[i],
                                            strides[i], dilations[i], dropout_rate1=dropout_rates1[i], dropout_rate2=dropout_rates2[i])
                for i in range(len(group_sizes))]
        self.residual_groups = nn.Sequential(*groups)

        output_dropout_rate = kwargs.get('output_dropout_rate', 0.5)
        if output_block is None:
            self.output_block = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(p=output_dropout_rate),
                nn.Linear(self.planes[-1] * block_type.expansion, num_outputs)
            )
        else:
            self.output_block = output_block(self.planes[-1] * block_type.expansion, num_outputs, dropout_rate=output_dropout_rate)

        self._initialize(zero_init_residual)

    def _make_residual_group1d(self, block_type, planes, kernel_size, blocks, stride=1, dilation=1, dropout_rate1=0.1, dropout_rate2=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block_type.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block_type.expansion))
        layers = []
        layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size,
                         stride=stride, dilation=dilation, downsample=downsample, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2))       
        self.inplanes = planes * block_type.expansion
        for _ in range(1, blocks):
            layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2))
        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        x = self.output_block(x)
        return x

class IMUResNetModel(nn.Module):
    def __init__(self, input_size, channels, output_size, dropout_rates, group_sizes):
        super(IMUResNetModel, self).__init__()
        self.resnet = ResNet1D(
            num_inputs=input_size,
            num_outputs=output_size,
            block_type=BasicBlock1D,
            group_sizes=group_sizes,
            base_plane=channels[0],
            output_block=FCOutputModule,
            input_dropout_rate=dropout_rates['input'],
            dropout_rates1=dropout_rates['block1'],
            dropout_rates2=dropout_rates['block2'],
            output_dropout_rate=dropout_rates['output'],
            fc_dim=1024  
        )
    def forward(self, x):
        return self.resnet(x)