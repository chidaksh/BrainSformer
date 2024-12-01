import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)


class UNet3DDecoder(nn.Module):
    def __init__(self, embed_dim=768, output_channels=4, base_channels=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_channels = output_channels

        # Initial projection to start decoding
        self.project1 = nn.Linear(embed_dim, base_channels * 8)  # Project to 3D shape
        self.project2 = nn.Linear(base_channels * 8, base_channels * 64)
        self.init_reshape = (base_channels * 64, 1, 1, 1)

        # Upsampling blocks
        self.up1 = self._upsample_block(base_channels * 64, base_channels * 32)
        self.up2 = self._upsample_block(base_channels * 32, base_channels * 16)
        self.up3 = self._upsample_block(base_channels * 16, base_channels * 8)
        self.up4 = self._upsample_block(base_channels * 8, base_channels * 4)
        self.up5 = self._upsample_block(base_channels * 4, base_channels * 2)
        self.up6 = self._upsample_block(base_channels * 2, base_channels)
        self.up7 = self._upsample_block(base_channels, base_channels // 2)

        # Final convolution to output channels
        self.final_conv = nn.Conv3d(base_channels // 2, output_channels, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels):
        """
        Upsample block with transposed convolution and residual refinement.
        """
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            ResidualBlock3D(out_channels, out_channels),
        )

    def forward(self, x):
        """
        Forward pass:
        x: (B, embed_dim)
        Returns:
        - mask: (B, output_channels, 128, 128, 128)
        """
        B = x.size(0)

        # Step 1: Initial projection and reshaping
        x = self.project1(x)  # (B, base_channels * 8 * 1 * 1 * 1)
        x = self.project2(x)
        x = x.view(B, *self.init_reshape)  # (B, base_channels * 64, 1, 1, 1)

        # Step 2: Hierarchical upsampling
        x = self.up1(x)  
        x = self.up2(x)  
        x = self.up3(x) 
        x = self.up4(x) 
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        # Step 3: Final convolution
        x = self.final_conv(x)  # (B, output_channels, 128, 128, 128)

        return x
