import torch
import torch.nn as nn
import torch.nn.functional as F


class SACI(nn.Module):
    def __init__(self, in_channels, n1_dilation, n2_dilation):
        super().__init__()

        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.dilated_conv_n1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=n1_dilation,
                                         dilation=n1_dilation)
        self.dilated_conv_n2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=n2_dilation,
                                         dilation=n2_dilation)

        self.conv1x1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.stacked_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

        self.conv1x1_3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        residual = x

        # First 1x1 conv
        out = self.conv1x1_1(x)

        # Dilated Convs
        out_n1 = self.dilated_conv_n1(out)
        out_n2 = self.dilated_conv_n2(out)

        # element-wise summation
        out = out_n1 + out_n2

        # Stacked conv and 1x1 conv output
        out = self.stacked_conv(out)
        out = self.conv1x1_3(out)

        # add and output with Relu
        out = F.relu(out + residual)
        return out


if __name__ == '__main__':
    # Example Usage
    in_channels = 128  # Example number of input channels
    n1_dilation = 1  # Example dilation rate for N1 dilated conv
    n2_dilation = 2  # Example dilation rate for N2 dilated conv
    batch_size = 4
    height = 64
    width = 64

    # create a dummy input
    input_tensor = torch.randn(batch_size, in_channels, height, width)

    # Instantiate SACI module
    saci_module = SACI(in_channels, n1_dilation, n2_dilation)

    # Forward propagation
    output_tensor = saci_module(input_tensor)

    # Check output shape
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)