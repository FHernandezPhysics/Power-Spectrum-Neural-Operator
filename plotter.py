import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

log_path = "test_log.txt" # Open a log file to write training progress
open(log_path, 'w').close()  # Clear the log file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loops = np.load('/gpfs/scratch/ferhernandez/training_data/results_combined_train.npy') # shape [10000,210,8]
valid_loops = np.load('/gpfs/scratch/ferhernandez/training_data/results_valid.npy') # shapes [1000,210,8]
test_loops = np.load('/gpfs/scratch/ferhernandez/training_data/results_test.npy')

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
    x_np = x_te.cpu().numpy()[:,0,:]
    yt_np = y_te.cpu().numpy()[:,0,:]
    yp_np = predictions.cpu().numpy()[:,0,:]

    diff = (predictions - y_te).flatten(1)
    ref = y_te.flatten(1)
    mse = torch.mean(diff ** 2, dim=-1)
    rel = torch.norm(diff, dim=-1) / torch.norm(ref, dim=-1)
    with open(log_path, 'a') as f:
        f.write(f"Average MSE={mse.mean().item()}, Average Relative Error={rel.mean().item()}\n")
        for i, (m,r) in enumerate(zip(mse.tolist(), rel.tolist())):
            f.write(f"Sample {i}: MSE={m}, Relative Error={r}\n")

fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(7, 7), sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
)
n = 20
exp_colors = plt.cm.Blues(np.linspace(0.45, 1.0, n))     # expected: blues
mod_colors = plt.cm.Oranges(np.linspace(0.45, 1.0, n))   # model: oranges

# Top panel: expected vs. model
for i in range(n):
    ax.loglog(np.abs(k_col[i]), np.abs(yt_np[i]), '-', color=exp_colors[i], lw=2, label=f'Expected {i}')
    ax.loglog(np.abs(k_col[i]), np.abs(yp_np[i]), '--', color=mod_colors[i], lw=1, label=f'Model {i}')
ax.grid(True, which='both', alpha=0.3)

# Bottom panel: fractional error
for i in range(n):
    ax_r.semilogx(np.abs(k_col[i]), np.abs(0.04 * (yp_np[i] - yt_np[i]) / x_np[i]), '-', color=exp_colors[i], lw=1, label=f'Ratio {i}')
ax_r.set_ylim(0, 0.02)
ax_r.set_xlim(1e-3, 50)
ax_r.set_xlabel('k')
ax_r.set_ylabel('|model/expected - 1|')
ax_r.grid(True, which='both', alpha=0.3)

plt.savefig('predictions_plot.png', dpi=300, bbox_inches='tight')
plt.close()