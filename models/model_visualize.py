from ASTGCN import make_model
import torch
from torchviz import make_dot

device = 'cpu'
A = torch.randn(134, 134)
model =  make_model(device,
                        nb_block=2,
                        in_channels=10,
                        K=3,
                        nb_chev_filter=64,
                        nb_time_filter=64,
                        time_strides=1,
                        adj_mx=A,
                        num_for_predict=24,
                        len_input=48,
                        num_of_vertices=134)

x = torch.randn(32, 134, 10, 48)

# 可视化计算图
y = model(x)
make_dot(y, params=dict(model.named_parameters())).view()
