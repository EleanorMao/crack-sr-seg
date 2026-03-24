"""
U-Net Model Definition for Crack Segmentation
Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upsampling + DoubleConv with skip connection"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net Model
    Args:
        in_channels: Input channels
        out_channels: Output channels
        features: Feature sizes per stage
    """

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        self.up1 = Up(features[3], features[2])
        self.up2 = Up(features[2], features[1])
        self.up3 = Up(features[1], features[0])

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.outc(x)
        return x


class DiceLoss(nn.Module):
    """Dice Loss"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Loss: BCE + Dice"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight

        # BCEWithLogitsLoss without pos_weight (will be handled in forward)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        # Handle pos_weight with correct device
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=predictions.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                predictions, targets, pos_weight=pos_weight, reduction='mean'
            )
        else:
            bce_loss = self.bce(predictions, targets)

        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def compute_iou(pred, target, threshold=0.5, smooth=1e-6):
    """Compute IoU (Intersection over Union)"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


def compute_dice_coeff(pred, target, threshold=0.5, smooth=1e-6):
    """Compute Dice Coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum()

    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice


def compute_pixel_accuracy(pred, target, threshold=0.5):
    """Compute Pixel Accuracy"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    correct = (pred == target).sum()
    total = target.numel()

    return correct.float() / total


if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = CombinedLoss(pos_weight=5.0)
    target = torch.zeros(1, 1, 256, 256)
    loss = criterion(y, target)
    print(f"Loss: {loss.item():.4f}")
