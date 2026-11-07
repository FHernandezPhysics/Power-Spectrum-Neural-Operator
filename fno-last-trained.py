import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import datetime

"""
This model is the most recently trained version. The model in fno.py
is the version we use for development, while this is an archive of the
model's state at the time of its last training.
"""

log_path = "training_log.txt"
open(log_path, 'w').close()  # Clear the log file

start = datetime.datetime.now()
msg = f"Start time: {start}"
with open(log_path, 'a') as f:
                f.write(msg + '\n')


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

def training_loop(model, optimizer, loss, x_tr, y_tr, x_va, y_va, n_epochs, log_no, batch_size):
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    tr_hist, va_hist = [], []
    best_va = best_tr = float('inf')
    btr = bva = ""

    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            l = loss(model(xb), yb)
            l.backward()
            optimizer.step()
            running += l.item() * xb.size(0)
        train_loss = running / len(x_tr)

        model.eval()
        with torch.no_grad():
            valid_loss = loss(model(x_va), y_va).item()

        tr_hist.append(train_loss)
        va_hist.append(valid_loss)

        print_bool = (epoch % log_no == 0 
                      or train_loss < best_tr 
                      or valid_loss < best_va
                      or epoch == n_epochs - 1)

        if train_loss < best_tr:
            best_tr = train_loss
            btr = f"Best Training Loss: {best_tr:.6f} at epoch {epoch}"

        if valid_loss < best_va:
            best_va = valid_loss
            bva = f"Best Validation Loss: {best_va:.6f} at epoch {epoch}"
            torch.save(model.state_dict(), 'fourier_model.pth')
            
        if print_bool:
            now = datetime.datetime.now()
            msg = f"Epoch {epoch}: {now}: Train Loss = {train_loss:.6f}, Valid Loss = {valid_loss:.6f}"
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

    return tr_hist, va_hist, btr, bva

n_epochs = 5000
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.MSELoss()
log_no = 100
batch_size = 100


tr_hist, va_hist, btr, bva = training_loop(model, optimizer, loss, x_tr, y_tr, x_va, y_va, n_epochs, log_no, batch_size)
torch.save(tr_hist, 'training_results.npy')
torch.save(va_hist, 'validation_results.npy')

end = datetime.datetime.now()
msg = f"End time: {end}"
dur = f"Duration: {end - start}"
with open(log_path, 'a') as f:
                f.write(msg + '\n')
                f.write(dur + '\n')
                f.write(btr + '\n')
                f.write(bva + '\n')