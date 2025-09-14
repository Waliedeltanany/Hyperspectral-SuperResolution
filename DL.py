import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from PIL import Image

# Load Data
print("Loading C:\\Users\\tanan\\OneDrive\\Desktop\\Super_R\\PaviaU.mat ...")
data = loadmat(r"C:\Users\tanan\OneDrive\Desktop\Super_R\PaviaU.mat")

print("Variables inside .mat:", list(data.keys()))
X = data['paviaU']  # (610, 340, 103)
print("Original data shape:", X.shape)

# Rearrange to (C, H, W)
X = np.transpose(X, (2, 0, 1))
print("Final data shape:", X.shape)

# Normalize
X = X / np.max(X)

# Make patches (dummy simple patching)
patch_size = 32
stride = 32
patches = []
for i in range(0, X.shape[1] - patch_size, stride):
    for j in range(0, X.shape[2] - patch_size, stride):
        patch = X[:, i:i+patch_size, j:j+patch_size]
        if patch.shape[1:] == (patch_size, patch_size):
            patches.append(patch)
patches = np.array(patches)
print("Total patches:", len(patches))

# LR and HR (simulate LR by downsampling)
hr_patches = patches
lr_patches = np.array([p[:, ::2, ::2] for p in patches])

# Torch dataset
hr_torch = torch.tensor(hr_patches, dtype=torch.float32)
lr_torch = torch.tensor(lr_patches, dtype=torch.float32)
train_ds = TensorDataset(lr_torch, hr_torch)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

# Model
class SimpleSRNet(nn.Module):
    def __init__(self, in_channels):
        super(SimpleSRNet, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, in_channels, 3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSRNet(in_channels=X.shape[0]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Save Hyperspectral Image (RGB)
def save_hsi_image(cube, path, bands=(30, 20, 10)):
    """
    cube: tensor (C, H, W)
    bands: tuple of 3 band indices to use as RGB
    """
    cube = cube.detach().cpu().numpy()
    if cube.shape[0] < 3:
        img = cube[0]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = Image.fromarray(img.astype(np.uint8))
    else:
        r, g, b = bands
        img = np.stack([
            cube[r],
            cube[g],
            cube[b]
        ], axis=-1)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = Image.fromarray(img.astype(np.uint8))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"Saved image: {path}")


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

for epoch in range(10):
    for step, (lr, hr) in enumerate(train_dl):
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")

    # PSNR
    with torch.no_grad():
        sr = model(lr)
        score = psnr(sr, hr)
        print(f"Epoch {epoch}, PSNR: {score:.2f}")

with torch.no_grad():
    lr, hr = next(iter(train_dl))
    lr, hr = lr.to(device), hr.to(device)
    sr = model(lr)

    save_hsi_image(lr[0], "results/lr.png")
    save_hsi_image(sr[0], "results/sr.png")
    save_hsi_image(hr[0], "results/hr.png")
