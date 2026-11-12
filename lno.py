import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import datetime

log_path = "lno_training_log.txt" # Open a log file to write training progress
open(log_path, 'w').close()  # Clear the log file

# Records start time
start = datetime.datetime.now()
msg = f"Start time: {start}"
with open(log_path, 'a') as f:
                f.write(msg + '\n')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Imports dataset
train_loops = np.load('training_data/results_combined_train.npy') # shape [10000,210,8]
valid_loops = np.load('training_data/results_valid.npy') # shapes [1000,210,8]
test_loops = np.load('training_data/results_test.npy')

# Pulls relevant data and converts to torch tensors
# each has shape [10000,1,210]
k_col = torch.from_numpy(train_loops[:,:,0]).to(device)
x_tr = torch.from_numpy(train_loops[:,:,1]).to(device).unsqueeze(1).float() # Training data
y_tr = torch.from_numpy(train_loops[:,:,4]).to(device).unsqueeze(1).float()

# each has shape [1000,1,210]
x_va = torch.from_numpy(valid_loops[:,:,1]).to(device).unsqueeze(1).float() # Validation data
y_va = torch.from_numpy(valid_loops[:,:,4]).to(device).unsqueeze(1).float()

x_te = torch.from_numpy(test_loops[:,:,1]).to(device).unsqueeze(1).float() # Test data
y_te = torch.from_numpy(test_loops[:,:,4]).to(device).unsqueeze(1).float()

# Defines the low rank layers for the LNO
class LowRankLayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.slen = x_tr.shape[-1]
        self.conv = nn.Conv1d(in_features, out_features, 1)
        self.phi = nn.Parameter(torch.randn(out_features, in_features, rank, self.slen, dtype=torch.float) 
                                / (in_features * rank))
        self.psi = nn.Parameter(torch.randn(out_features, in_features, rank, self.slen, dtype=torch.float) 
                                / (in_features * rank))

    def forward(self, x):
        batch_size, _, signal_length = x.shape
        coef = torch.einsum('oirn,bin->boir', self.psi, x) / signal_length
        lno = torch.einsum('oirm,boir->bom', self.phi, coef)
        lin = self.conv(x)
        return torch.nn.functional.relu(lin + lno)
        

width,rank = 32, 16

# Defines the LNO model
model = nn.Sequential(
    nn.Conv1d(1, width, 1), # input layer, does lifting
    LowRankLayer(width, width, rank), # low rank layer block
    LowRankLayer(width, width, rank), 
    LowRankLayer(width, width, rank), 
    LowRankLayer(width, width, rank),
    nn.Conv1d(width, 128, 1), # linear layer
    nn.ReLU(),
    nn.Conv1d(128, 1, 1) # output layer
).to(device)

# Defines the training loop for the model
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

        # Records training and validation loss history
        tr_hist.append(train_loss)
        va_hist.append(valid_loss)

        # Print if epoch mod10 = 0 or if train/val loss pb and on final epoch
        print_bool = (epoch % log_no == 0 
                      or train_loss < best_tr 
                      or valid_loss < best_va
                      or epoch == n_epochs - 1)

        # Saves best training loss
        if train_loss < best_tr:
            best_tr = train_loss
            btr = f"Best Training Loss: {best_tr:.6f} at epoch {epoch}"

        # Saves best validation loss and model state
        if valid_loss < best_va:
            best_va = valid_loss
            bva = f"Best Validation Loss: {best_va:.6f} at epoch {epoch}"
            torch.save(model.state_dict(), 'fourier_model.pth')

        # Print training progress to log file
        if print_bool:
            now = datetime.datetime.now()
            msg = f"Epoch {epoch}: {now}: Train Loss = {train_loss:.6f}, Valid Loss = {valid_loss:.6f}"
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

    return tr_hist, va_hist, btr, bva # Outputs metrics

n_epochs = 5000
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.MSELoss()
log_no = 100
batch_size = 100

# Runs the training loop and saves the training and validation loss history
tr_hist, va_hist, btr, bva = training_loop(model, optimizer, loss, x_tr, y_tr, x_va, y_va, n_epochs, log_no, batch_size)
torch.save(tr_hist, 'training_results.npy')
torch.save(va_hist, 'validation_results.npy')
# Runs the training# Runs the training loop and saves the training and validation loss history
tr_hist, va_hist, btr, bva = training_loop(model, optimizer, loss, x_tr, y_tr, x_va, y_va, n_epochs, log_no, batch_size)
torch.save(tr_hist, 'training_results.npy')
torch.save(va_hist, 'validation_results.npy')

# Records end time and duration of training
end = datetime.datetime.now()
msg = f"End time: {end}"
dur = f"Duration: {end - start}"
with open(log_path, 'a') as f:
                f.write(msg + '\n')
                f.write(dur + '\n')
                f.write(btr + '\n') # Records best training loss
                f.write(bva + '\n') # best validation loss loop and saves the training and validation loss history
tr_hist, va_hist, btr, bva = training_loop(model, optimizer, loss, x_tr, y_tr, x_va, y_va, n_epochs, log_no, batch_size)
torch.save(tr_hist, 'training_results.npy')
torch.save(va_hist, 'validation_results.npy')

# Records end time and duration of training
end = datetime.datetime.now()
msg = f"End time: {end}"
dur = f"Duration: {end - start}"
with open(log_path, 'a') as f:
                f.write(msg + '\n')
                f.write(dur + '\n')
                f.write(btr + '\n') # Records best training loss
                f.write(bva + '\n') # best validation loss
# Records end time and duration of training
end = datetime.datetime.now()
msg = f"End time: {end}"
dur = f"Duration: {end - start}"
with open(log_path, 'a') as f:
                f.write(msg + '\n')
                f.write(dur + '\n')
                f.write(btr + '\n') # Records best training loss
                f.write(bva + '\n') # best validation loss loop and saves the training and validation loss history
tr_hist, va_hist, btr, bva = training_loop(model, optimizer, loss, x_tr, y_tr, x_va, y_va, n_epochs, log_no, batch_size)
torch.save(tr_hist, 'training_results.npy')
torch.save(va_hist, 'validation_results.npy')

# Records end time and duration of training
end = datetime.datetime.now()
msg = f"End time: {end}"
dur = f"Duration: {end - start}"
with open(log_path, 'a') as f:
                f.write(msg + '\n')
                f.write(dur + '\n')
                f.write(btr + '\n') # Records best training loss
                f.write(bva + '\n') # best validation loss