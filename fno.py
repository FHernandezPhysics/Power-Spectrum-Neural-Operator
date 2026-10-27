import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FourierLayer(nn.Module):
    def __init__(self, n, kmax):
        super().__init__()

        # weight matrix and kernel tensor
        self.W = nn.Parameter(torch.randn(n, n))
        self.R = nn.Parameter(torch.randn(kmax, n, n))

    def forward(self, x):
        x_fft    = torch.fft.fft(x, dim=-1) # take fourier transform of input
        Wx = torch.matmul(x, self.W) # multiply input with weight matrix
        Rx_fft = torch.einsum("knm,...m->...kn", self.R, x_fft) #multiply input with kernel tensor in Fourier space
        Rx = torch.fft.ifft(Rx_fft, dim=-1).real # inverse fourier transform to get back to real space

        return Wx + Rx # return layer output

# one layer of the model with ReLU activation
fno_block = nn.Sequential(
    FourierLayer(n=100, kmax=100),
    nn.ReLU())

lr = 1e-3
optimizer = torch.optim.Adam(fno_block.parameters(), lr=lr)
loss = nn.MSELoss()
n_epochs = 1000

# dummy data as a placeholder
inpDummy = torch.randn(100, device=device)  # Dummy input tensor, n=100

weightMat = torch.randn(100, 100, device=device)  # Dummy weight matrix, n*n
kerMatR = torch.randn(100, 100, 100, device=device)  # Dummy kernel matrix, kmax*n*n

# basic training loop for the block
for epoch in range(n_epochs):
    optimizer.zero_grad()
    output = fno_block(inpDummy)
    target = torch.matmul(weightMat, inpDummy)  # Dummy target for loss calculation
    loss_value = loss(output, target)
    loss_value.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.item()}")
