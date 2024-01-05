import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm 
from layer import *

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
                 dropout=0,
                 device='cpu',
                 batch_first=True
                 ):
        super(LSTM, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.device      = device
        self.dropout     = dropout
        self.batch_first = batch_first
        self.model = nn.Sequential(
            nn.LSTM(input_size=self.input_size, 
                   hidden_size  = self.hidden_size,
                   num_layers   = self.num_layers,
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
    

class GRU(nn.Module):
    def __init__(self, input_size, 
                 hidden_size, 
                 output_size,
                 num_layers=1,
                 dropout=0,
                 device='cpu',
                 batch_first=True
                 ):
        super(GRU, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.device      = device
        self.dropout     = dropout
        # batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) 
        self.batch_first = batch_first
        self.model = nn.Sequential(
            nn.GRU(input_size=self.input_size, 
                   hidden_size  = self.hidden_size,
                   num_layers   = self.num_layers,
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
    


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
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
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
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
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入维度，即特征数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            # 确定每一层的输入通道数
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 num_channels, 
                 kernel_size, 
                 dropout, 
                 device='cuda'):
        """
        Temporal Convolutional Network.

        Args:
            input_size: int，输入特征数;
            output_size: int，输出特征数, 即输出维度，回归为1，分位数为num_quantiles;
            num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
            kernel_size: int, 卷积核尺寸
            dropout: float, drop_out比率
            device: str, cuda or cpu
        """
        super(TCN, self).__init__()
        self.input_size = input_size
        self.device     = device
        self.tcn        = TemporalConvNet(self.input_size, 
                                   num_channels, 
                                   kernel_size, 
                                   dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def init(self, W_var):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_size, seq_len)
        :return: size of (Batch, output_size)
        """
        # x: [batch_size, input_len, num_features]
        # x = x.reshape(x.shape[0], self.input_size, -1)
        x = x.permute(0, 2, 1)
        x  = x.to(self.device)
        y1 = self.tcn(x)

        return self.linear(y1[:, :, -1])



class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4


# Copied from https://github.com/nnzhan/MTGNN
class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, 
                 predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, 
                 node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, 
                 skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, 
                 propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        """
        Initializes a GTNet model.

        Args:
            gcn_true (bool): Whether to use graph convolutional networks (GCN) or not.
            buildA_true (bool): Whether to build adjacency matrix A or use a predefined one.
            gcn_depth (int): The number of GCN layers.
            num_nodes (int): The number of nodes in the graph.
            device (torch.device): The device to run the model on.
            predefined_A (torch.Tensor, optional): A predefined adjacency matrix. Defaults to None.
            static_feat (torch.Tensor, optional): Static features for each node. Defaults to None.
            dropout (float, optional): Dropout rate. Defaults to 0.3.
            subgraph_size (int, optional): Size of the subgraph. Defaults to 20.
            node_dim (int, optional): Dimension of the node features. Defaults to 40.
            dilation_exponential (int, optional): Exponential factor for dilation. Defaults to 1.
            conv_channels (int, optional): Number of channels in the convolutional layers. Defaults to 32.
            residual_channels (int, optional): Number of channels in the residual layers. Defaults to 32.
            skip_channels (int, optional): Number of channels in the skip connections. Defaults to 64.
            end_channels (int, optional): Number of channels in the final convolutional layer. Defaults to 128.
            seq_length (int, optional): Length of the input sequence. Defaults to 12.
            in_dim (int, optional): Input dimension. Defaults to 2.
            out_dim (int, optional): Output dimension. Defaults to 12.
            layers (int, optional): Number of layers. Defaults to 3.
            propalpha (float, optional): Propagation alpha for the graph convolutional layers. Defaults to 0.05.
            tanhalpha (int, optional): Tanh alpha for the graph constructor. Defaults to 3.
            layer_norm_affline (bool, optional): Whether to use affine transformations in the layer normalization. Defaults to True.
        
        Returns:
            None
        """
        super(gtnet, self).__init__()
        self.gcn_true       = gcn_true
        self.buildA_true    = buildA_true
        self.num_nodes      = num_nodes
        self.dropout        = dropout
        self.predefined_A   = predefined_A
        self.filter_convs   = nn.ModuleList()
        self.gate_convs     = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs     = nn.ModuleList()
        self.gconv1         = nn.ModuleList()
        self.gconv2         = nn.ModuleList()
        self.norm           = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
