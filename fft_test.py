import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

train_loops = np.load('/home/fhsci/Cosmology/training_data/results_combined_train.npy') # shape [10000,210,8]
valid_loops = np.load('/home/fhsci/Cosmology/training_data/results_valid.npy') # shapes [1000,210,8]
test_loops = np.load('/home/fhsci/Cosmology/training_data/results_test.npy')

k = test_loops[0,:,0].astype(np.float64)
x = test_loops[:,:,1].astype(np.float64)
y = test_loops[:,:,4].astype(np.float64)

n = y.shape[-1]

# 0) roundtrip control, should be ~1e-15
Y = np.fft.rfft(y, axis=-1)
y_rt = np.fft.irfft(Y, n=n, axis=-1)
print("max roundtrip rel err:", np.max(np.abs(y_rt / y - 1)))
print("max abs roundtrip err:", np.max(np.abs(y_rt - y)))
print("y min, max:", y.min(), y.max())

# 1) perturb every Fourier mode's amplitude by ~1e-6 (relative), inverse transform
rng = np.random.default_rng(0)
Yp = Y * (1 + 1e-6 * rng.standard_normal(Y.shape))
y_p = np.fft.irfft(Yp, n=n, axis=-1)

err_rel = np.abs(y_p / y - 1)
err_abs = np.abs(y_p - y)

# 2) summarize by position along k (median over samples), plus a plot
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
ax[0].semilogx(k, np.median(err_rel, axis=0)); ax[0].set_yscale("log"); ax[0].set_ylabel("median |rel err|")
ax[1].semilogx(k, np.median(err_abs, axis=0)); ax[1].set_yscale("log"); ax[1].set_ylabel("median |abs err|")
ax[1].set_xlabel("k")
plt.savefig("fft_edge_test.png", dpi=150)