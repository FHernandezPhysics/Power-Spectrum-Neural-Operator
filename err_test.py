import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

log_path = "test_log.txt" # Open a log file to write training progress
open(log_path, 'w').close()  # Clear the log file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loops = np.load('training_data/results_combined_train.npy') # shape [10000,210,8]
valid_loops = np.load('training_data/results_valid.npy') # shapes [1000,210,8]
test_loops = np.load('training_data/results_test.npy')

# each has shape [10000,1,210]
k_col = torch.from_numpy(train_loops[:,:,0]).to(device)
x_tr = torch.from_numpy(train_loops[:,:,1]).to(device).unsqueeze(1).float()
y_tr = torch.from_numpy(train_loops[:,:,4]).to(device).unsqueeze(1).float()

# each has shape [1000,1,210]
x_va = torch.from_numpy(valid_loops[:,:,1]).to(device).unsqueeze(1).float()
y_va = torch.from_numpy(valid_loops[:,:,4]).to(device).unsqueeze(1).float()

x_te = torch.from_numpy(test_loops[:,:,1]).to(device).unsqueeze(1).float()
y_te = torch.from_numpy(test_loops[:,:,4]).to(device).unsqueeze(1).float()

class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, modes):
        super(FourierLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.kernel = nn.Parameter((1 / (in_features * out_features)) * 
                                   torch.randn(in_features, out_features, modes,
                                                dtype=torch.cfloat))

    def forward(self, x):
        batch_size, _, signal_length = x.shape
        x_ft = torch.fft.rfft(x, dim=-1) # take fft of input
        Rft = torch.einsum('bin,ion->bon', x_ft[:, :, :self.modes], self.kernel) # multiply input by the kernel
        x_if = torch.fft.irfft(Rft, n=signal_length, dim=-1) # take inverse fft to get fourier output
        Wx = torch.einsum('oi,bin->bon', self.weight, x) # multiply input by weight to get linear output
        return torch.nn.functional.relu(Wx + x_if) # add linear and fourier outputs and apply relu

width,modes = 32, 16

model = nn.Sequential(
    nn.Conv1d(1, width, 1), # input layer
    FourierLayer(width, width, modes), # fourier layer
    FourierLayer(width, width, modes), # fourier layer
    FourierLayer(width, width, modes), # fourier layer
    FourierLayer(width, width, modes), # fourier layer
    nn.Conv1d(width, 128, 1), # linear layer
    nn.ReLU(),
    nn.Conv1d(128, 1, 1) # output layer
).to(device)

model.load_state_dict(torch.load('fourier_model.pth'))
model.eval()

with torch.no_grad():
    predictions = model(x_te)
    mse = torch.mean((predictions - y_te) ** 2)
    rel = torch.norm(predictions - y_te, dim=-1) / torch.norm(y_te, dim=-1)
    msg1 = f"Test MSE: {mse.item()}" 
    msg2 = f"Test Relative Error: {rel.mean().item()}"
    with open(log_path, 'a') as f:
        f.write(msg1 + '\n')
        f.write(msg2 + '\n')