# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import scaled_Laplacian, cheb_polynomial
from torch.nn.utils import weight_norm  

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
    

class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE)) # (T,)
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))  # (F, T)
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))  # (F,)
        # self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        # self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, N, N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        # S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(product, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        # self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        # self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''


        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        # E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T) # 原版

        E_normalized = F.softmax(product, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


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
        # self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # self.relu = nn.ReLU()
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
        # print(f'block: x.shape: {x.shape}')
        out = self.net(x)
        # res = x if self.downsample is None else self.downsample(x)
        # return self.relu(out + res)

        return out
    
class TCN2d(nn.Module):
    def __init__(self, 
                 input_size  : int   = 10,
                 output_size : int   = 64,
                 num_channels: list  = [64, 128, 256],
                 kernel_size : int   = 3,
                 dropout     : float = 0.2,
                 device      : str   = 'cpu'):
        super(TCN2d, self).__init__()
        self.device = device

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = output_size if i == num_levels - 1 else num_channels[i]
            # print(f'i: {i}, dilation: {dilation_size}, in_channels: {in_channels}, out_channels: {out_channels}')
            layers += [TemporalBlock2d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        # self.linear  = nn.Linear(num_channels[-1], output_size)
        # self.final_conv = nn.Conv2d(num_channels[-1], output_size, kernel_size=(1, kernel_size), stride=(1, 1))

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_nodes, num_features, num_time_steps]
        """

        x = x.permute(0, 2, 1, 3)  # x: [bs, n, f, t] -> [bs, f, n, t]
        # print(f'tcn input: x.shape: {x.shape}')
        x = x.to(self.device)
        x = self.network(x)  # [bs, f, n, t]
        # print(f'tcn network output: x.shape: {x.shape}')
        # x = self.linear(x.permute(0, 2, 3, 1)) # [bs, num_channels[-1], n, t] -> [bs, n, t, num_channels[-1]] -> [bs, n, t, out_channels]
        # print(f'tcn linear output: x.shape: {x.shape}')

        # x = self.final_conv(x)  # [bs, num_channels[-1], n, t] -> [bs, out_channels, n, t]

        return x  # (b, F, N, T)

class GTCN_block(nn.Module):
    def __init__(self, 
                 DEVICE, 
                 in_channels, 
                 K, 
                 kernel_size,
                 nb_chev_filter, 
                 nb_time_filter,
                 output_time_steps,
                 tcn_channels: list, 
                 cheb_polynomials, 
                 num_of_vertices, 
                 num_of_timesteps,
                 time_strides: int = 1):
        """
        Initializes an instance of the GTCN_block class.

        Parameters:
        - DEVICE: The device on which the computation will be performed.
        - in_channels: The number of input channels. number of features.
        - K: The number of Chebyshev polynomials.
        - nb_chev_filter: The number of Chebyshev filters.
        - nb_time_filter: The number of time filters.
        - time_strides: The stride for the time convolution.
        - cheb_polynomials: The Chebyshev polynomials.
        - num_of_vertices: The number of vertices.
        - num_of_timesteps: The number of timesteps.

        Returns:
        - None
        """

        super(GTCN_block, self).__init__()
        self.TAt           = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt           = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        # self.time_conv     = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        # # self.time_conv2  = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 2))
        self.tcn           = TCN2d(nb_chev_filter, nb_time_filter, num_channels=tcn_channels, kernel_size=kernel_size, dropout=0.2, device=DEVICE)
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln            = nn.LayerNorm(nb_time_filter)  #需要将channel放到最后一个维度上
        # self.bn            = nn.BatchNorm2d(nb_time_filter)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape  # # (b, N, F, T)
        # print(f'GTCN block input x.shape: {x.shape}')

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        # (b N, F, T)

        # SAt
        spatial_At = self.SAt(x_TAt)  # (b, N, N)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b, N, F, T)，这里的 F 已经变为nb_chev_filter，下同
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        # time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T)

        # tcn input: [batch_size, num_nodes, num_features, num_time_steps]
        time_conv_output = self.tcn(spatial_gcn)  # (b, F, N, T) 

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)

        # # breakpoint()

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1) 
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)
        # print(f'x_residual shape: {x_residual.shape}')

        return x_residual


class GTCN(nn.Module):
    def __init__(self, 
                 adj_matrix,
                 DEVICE, 
                 nb_block, 
                 in_channels, 
                 K, 
                 nb_chev_filter, 
                 nb_time_filter,
                 kernel_size, 
                 input_time_steps,
                 output_time_steps, 
                 tcn_channels, 
                 num_of_vertices):
        """
        Initialize the GTCN model.

        Parameters:
            adj_matrix (np.ndarray): The adjacency matrix of the graph.
            DEVICE (str): The device to be used for computation.
            nb_block (int): The number of blocks in the GTCN model.
            in_channels (int): The number of input channels.
            K (int): The number of Chebyshev polynomials to be used.
            nb_chev_filter (int): The number of Chebyshev filters.
            kernel_size (int): The size of the kernel.
            input_time_steps (int): The number of input time steps.
            output_time_steps (int): The number of output time steps.
            cheb_polynomials (List[torch.Tensor]): The list of Chebyshev polynomials.
            tcn_channels (List[int]): The list of channels in the TCN layers.
            num_of_vertices (int): The number of vertices in the graph.

        Returns:
            None
        """

        super(GTCN, self).__init__()

        self.device = DEVICE
        L_tilde = scaled_Laplacian(adj_matrix)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
        self.model = nn.ModuleList(
            [GTCN_block(DEVICE, in_channels, K, kernel_size, nb_chev_filter, nb_time_filter, \
                        output_time_steps, tcn_channels, cheb_polynomials, num_of_vertices, input_time_steps)]
            )

        self.model.extend(
            [GTCN_block(DEVICE, nb_time_filter, K, kernel_size, nb_chev_filter, nb_time_filter, \
                        output_time_steps, tcn_channels, cheb_polynomials, num_of_vertices, input_time_steps)
            for _ in range(nb_block - 1)]
        )

        self.final_conv = nn.Conv2d(input_time_steps, output_time_steps, kernel_size=(1, nb_time_filter))

        # self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self.init_weights()

    def init_weights(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''

        x = x.to(self.device)
        model = self.model.to(self.device)

        for i, block in enumerate(model):
            x = block(x)
            # print(f'i: {i}, x.shape: {x.shape}')

        # print(f'TCN before output: x.shape = {x.shape}')
        # breakpoint()
        # x: (b, N, F, T)
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        # print(f'gtcn output shape: {output.shape}')
        # breakpoint()

        return output
