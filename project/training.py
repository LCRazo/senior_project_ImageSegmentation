import torch
from dataset.dataset import TrainMosDataset
import torch.nn as nn                      
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from unet.unet_model import UNet as unet  # Import your U-Net model implementation
import argparse

#similarity measure to evaluate segmentation performance. measures overlap between two masks
def dice_coefficient(predicted, target, smooth=1e-6):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted * target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predicted, target):
        # Implementation for custom loss function. Defines computation  
        # 1 converts measure into loss that can be minimized, measure of similarity wanted to maximize 
        dice = 1 - dice_coefficient(predicted, target)
        return dice

def main():
    # Define hyperparameters

    parser = argparse.ArgumentParser(description='U-Net Training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    # parser.add_argument('--gcn', type=str, default='hogcn')
    # parser.add_argument('--np_ratio', type=float, default=0.005)
    # parser.add_argument('--k_ratio', type=float, default=0.5)
    # parser.add_argument('--os', type=int, default=8)

    parser.add_argument('--train_list', type=str, default='train_new0',
                        choices=('train_new0', 'train_new1', 'train_new2',
                                 'train_new3', 'train_new4'))
    # parser.add_argument('--valid_list', type=str, default='valid_new0',
    #                     choices=('valid_new0', 'valid_new1', 'valid_new2',
    #                              'valid_new3', 'valid_new4'))
    # parser.add_argument('--datadir', type=str, default='')

    args = parser.parse_args()


    # Loading Datatset
    print('======> Loading train datasets on:', args.train_list)
    train_dataset = TrainMosDataset(args.train_list, datadir=args.datadir, args=args, transforms=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the U-Net model
    #3 gray scale, and 3 regions of interest (background, lung, covid region)
    model = unet(in_channels=3, out_channels=3, )
    print('using unet......')
    # # ***code from template to use as reference***
    # model = unet(n_channels=3, n_classes=1, embedded_module=args.embedded_module, gcn=args.gcn, np_ratio=args.np_ratio, k_ratio=args.k_ratio, os=args.os)
    # print('using unet......')

    # Define loss function and optimizer (We need Dice Loss)
    # criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()
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


if __name__ == '__main__':
    main()