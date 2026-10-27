import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FourierLayer(nn.Module):
    def __init__(self, n, kmax):
        super().__init__()

        self.W = nn.Parameter(torch.randn(n, n))
        self.R = nn.Parameter(torch.randn(kmax, n, n))

    def forward(self, x):
        x_fft    = torch.fft.fft(x, dim=-1)
        Wx = torch.matmul(x, self.W)
        Rx_fft = torch.einsum("knm,...m->...kn", self.R, x_fft)
        Rx = torch.fft.ifft(Rx_fft, dim=-1).real

        return Wx + Rx

fno_block = nn.Sequential(
    FourierLayer(n=100, kmax=100),
    nn.ReLU())

lr = 1e-3
optimizer = torch.optim.Adam(fno_block.parameters(), lr=lr)
loss = nn.MSELoss()
n_epochs = 1000


inpDummy = torch.randn(100, device=device)  # Dummy input tensor, n=100

weightMat = torch.randn(100, 100, device=device)  # Dummy weight matrix, n*n
kerMatR = torch.randn(100, 100, 100, device=device)  # Dummy kernel matrix, kmax*n*n

for epoch in range(n_epochs):
    optimizer.zero_grad()
    output = fno_block(inpDummy)
    target = torch.matmul(weightMat, inpDummy)  # Dummy target for loss calculation
    loss_value = loss(output, target)
    loss_value.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.item()}")
