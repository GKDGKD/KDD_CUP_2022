import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from torch.nn.utils import weight_norm 

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding

        Args:
            x: [batch_size, num_channel, num_time_steps+padding]
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1   = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2   = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)  # [batch_size, output_channel, seq_len]
        res = x if self.downsample is None else self.downsample(x)
        # breakpoint()

        return self.relu(out + res)

class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        裁剪多余的padding。

        Args:
            x: [batch_size, num_nodes, num_features, num_time_steps]
        """
        return x[:, :, :, : -self.chomp_size].contiguous()
    
class TemporalBlock2d(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock2d, self).__init__()
        padding  = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size=(1, kernel_size),
                                           stride=stride, padding=(0, padding), dilation=dilation))
        self.chomp1   = Chomp2d(padding)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, kernel_size=(1, kernel_size),
                                           stride=stride, padding=(0, padding), dilation=dilation))
        self.chomp2   = Chomp2d(padding)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_nodes, num_features, num_time_steps]
        """
        print(f'block: x.shape: {x.shape}')
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
class TCN2d(nn.Module):
    def __init__(self, 
                 input_size: int = 10, 
                 output_size: int = 64,
                 num_channels: list = [64, 128, 256], 
                 kernel_size: int = 3, 
                 dropout: float = 0.2,
                 device: str = 'cpu'):
        super(TCN2d, self).__init__()
        self.device = device

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            print(f'i: {i}, dilation: {dilation_size}, in_channels: {in_channels}, out_channels: {out_channels}')
            layers += [TemporalBlock2d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear  = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_nodes, num_features, num_time_steps]
        """

        x = x.permute(0, 2, 1, 3)  # x: [bs, n, f, t] -> [bs, f, n, t]
        print(f'tcn input: x.shape: {x.shape}')
        x = x.to(self.device)
        x = self.network(x)  # [bs, f, n, t] -> [bs, num_channels[-1], n, t]
        print(f'tcn network output: x.shape: {x.shape}')
        x = self.linear(x.permute(0, 2, 3, 1)) # [bs, num_channels[-1], n, t] -> [bs, n, t, num_channels[-1]] -> [bs, n, t, out_channels]
        print(f'tcn linear output: x.shape: {x.shape}')
        return x.permute(0, 3, 1, 2)  # [bs, n, t, out_channels] -> [bs, out_channels, n, t]



batch_size       = 64
num_of_timesteps = 48
num_of_nodes     = 132
num_of_features  = 10
num_filters      = 64
num_time_filter  = 64
time_strides     = 1
kernel_size      = 3
stride           = 1
dilation         = 1
num_channels     = [16, 32, 64]

# layers = []
# num_layers = len(num_channels)
# for i in range(num_layers):
#     dilation_size = 2 ** i
#     in_channels = num_of_features if i == 0 else num_channels[i - 1]
#     out_channels = num_channels[i]
#     print(f'i: {i}, dilation: {dilation_size}, in_channels: {in_channels}, out_channels: {out_channels}')
#     layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride, dilation_size, padding=(kernel_size - 1)*dilation_size)]

# net = nn.Sequential(*layers)

# xx = torch.randn(batch_size, num_of_features, num_of_timesteps)
# print(f'xx.shape: {xx.shape}')

# xx2 = net(xx)
# print(f'xx2.shape: {xx2.shape}')

x = torch.randn(batch_size, num_of_nodes, num_of_features, num_of_timesteps)
print(f'x.shape: {x.shape}')

block1 = TemporalBlock2d(num_of_features, num_time_filter, kernel_size, stride, dilation=1)
x1 = block1(x.permute(0, 2, 1, 3)) # x: [bs, n, f, t] -> [bs, f, n, t] -> [bs, f, n, t]
print(f'x1.shape: {x1.shape}')

num_channels = [16, 32, 64]
kernel_size = 3
tcn = TCN2d(num_of_features, num_time_filter, num_channels, kernel_size)
x2 = tcn(x)
print(f'x2.shape: {x2.shape}')