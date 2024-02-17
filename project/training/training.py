import torch
import torch.nn as nn                      
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from your_dataset_module import YourDatasetClass
from unet_model import UNet  # Import your U-Net model implementation
import argparse

# Define hyperparameters
parser = argparse.ArgumentParser(description='U-Net Training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
args = parser.parse_args()

# Initialize the dataset and data loader
print('======> Loading train datasets on:', args.train_list)
train_dataset = YourDatasetClass(train=True, transform=ToTensor())  # Modify this according to your dataset implementation
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# Initialize the U-Net model
model = UNet()  # Instantiate your U-Net model

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(args.num_epochs):
    running_loss = 0.0
    for images, masks in train_loader:
        # Move images and masks to the appropriate device
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss for the epoch
    print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

print('Finished Training')
