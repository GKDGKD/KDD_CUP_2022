import torch
from models.GTCN import GTCN
from models.ASTGCN import make_model
from utils import get_adjency_matrix, get_normalized_adj

location_path = "./data/sdwpf_baidukddcup2022_turb_location.CSV"

A      = get_adjency_matrix(location_path, 1000)
A_wave = torch.from_numpy(get_normalized_adj(A)).to(torch.float32)
x = torch.randn(64, 134, 10, 48)

# GTCN
# model = GTCN(A_wave, 'cpu', 2, 10, 3, 64, 64, 3, 48, 24, [128, 256, 512, 256], 134)
# torch.onnx.export(model, x, "./models/GTCN.onnx", \
#                   export_params=True,verbose=True, input_names=['x'], output_names=['out'])

# ASTGCN
model = make_model('cpu', 2, 10, 3, 64, 64, 1, A_wave, 24, 48, 134)
torch.onnx.export(model, x, "./models/ASTGCN.onnx", \
                  export_params=True,verbose=True, input_names=['x'], output_names=['out'])