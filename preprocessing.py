import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Step 1: Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./Users/georg/OneDrive/covid19/covid100', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./Users/georg/OneDrive/covid19/covid100', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Data Checking (Optional)
# Print sample data
for images, labels in train_loader:
    print('Image batch dimensions before flattening:', images.size())  # Print original image dimensions
    break  # Break after printing the first batch

# Step 3: Data Preprocessing - Convert from 3D to 2D
# Flatten the images
num_features = images.size(1) * images.size(2) * images.size(3)
images = images.view(-1, num_features)

print('Image batch dimensions after flattening:', images.size())  # Print flattened image dimensions

# Step 4: Further Processing
# Additional preprocessing steps can be added here if needed