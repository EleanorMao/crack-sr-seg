"""
SRCNN Model Definition
Super-Resolution Convolutional Neural Network
Paper: Learning a Deep Convolutional Network for Image Super-Resolution (ECCV 2014)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    SRCNN Architecture:
    - Conv1: 9x9, 64 filters, ReLU
    - Conv2: 1x1, 32 filters, ReLU
    - Conv3: 5x5, 3 filters (output)
    """

    def __init__(self, num_channels=3, num_features=64):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(num_features, num_features // 2, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(num_features // 2, num_channels, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


class ImprovedSRCNN(nn.Module):
    """
    Improved SRCNN with:
    - Larger receptive field (first layer 9x9, last layer 5x5)
    - Residual connection for better gradient flow
    - Deeper non-linear mapping (3 middle layers)
    - No BatchNorm (BN can hurt SR quality by smoothing details)
    """

    def __init__(self, num_channels=3, num_features=64):
        super(ImprovedSRCNN, self).__init__()

        # Feature extraction (large kernel for receptive field)
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4)

        # Non-linear mapping
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Reconstruction
        self.conv5 = nn.Conv2d(num_features, num_channels, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = x  # Residual connection

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        # Residual learning: output = residual + input
        return x + identity


class ImprovedSRCNN_BN(nn.Module):
    """
    Improved SRCNN with BatchNorm:
    - Same architecture as ImprovedSRCNN but with BatchNorm layers
    - BatchNorm can help training stability but may affect SR quality
    """

    def __init__(self, num_channels=3, num_features=64):
        super(ImprovedSRCNN_BN, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(num_features)

        # Non-linear mapping
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)

        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features)

        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features)

        # Reconstruction
        self.conv5 = nn.Conv2d(num_features, num_channels, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)

        return x + identity


class ImprovedSRCNN_All3x3(nn.Module):
    """
    Improved SRCNN with all 3x3 kernels:
    - All convolution layers use 3x3 kernels (instead of 9x9 and 5x5)
    - Deeper network to compensate for smaller receptive field
    - Residual connection preserved
    """

    def __init__(self, num_channels=3, num_features=64):
        super(ImprovedSRCNN_All3x3, self).__init__()

        # Feature extraction (3x3 instead of 9x9)
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)

        # Non-linear mapping (more layers to build receptive field)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Reconstruction (3x3 instead of 5x5)
        self.conv7 = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = x

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)

        return x + identity


def compute_psnr(img1, img2):
    """Compute PSNR (Peak Signal-to-Noise Ratio)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM (Structural Similarity Index)"""
    channel = img1.size(1)

    window = create_window(window_size, channel).to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def create_window(window_size, channel):
    """Create Gaussian window for SSIM computation"""
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


if __name__ == '__main__':
    model = SRCNN()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
