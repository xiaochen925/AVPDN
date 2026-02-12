import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)
        # transpose
        x = x.transpose(1, 2).contiguous()
        # flatten
        x = x.view(b, -1, h, w)

        return x


class AFIA(nn.Module):
    def __init__(self, in_channels, groups):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_channels)  # Assuming layer norm is applied over channels

        self.conv1x1_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.deconv3x3_q = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.conv1x1_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.deconv3x3_k = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.conv1x1_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.deconv3x3_v = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.channel_shuffle = ChannelShuffle(groups)

        self.stacked_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

        self.conv1x1_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.layer_norm2 = nn.LayerNorm(in_channels)

    def depthwise_spatial_attention(self, Q, K):
        """Calculates the Depthwise Spatial Attention Map between Q and K.
         Args:
          Q: Tensor of shape (B, C, H, W)
          K: Tensor of shape (B, C, H, W)
         Returns:
            A spatial attention map with shape (B, H, W)
        """

        B, C, H, W = Q.shape
        Q = Q.permute(0, 2, 3, 1).reshape(B, H * W, C)  # B, HW, C
        K = K.permute(0, 2, 3, 1).reshape(B, H * W, C)  # B, HW, C

        attention_map = torch.matmul(Q, K.transpose(1, 2))  # B, HW, HW

        attention_map = F.softmax(attention_map / (C ** 0.5), dim=-1)  # B, HW, HW

        return attention_map

    def spatial_wise_self_attention(self, V, attention_map):
        """
        Apply spatial-wise self-attention to value feature map
         Args:
            V: Tensor of shape (B, C, H, W)
            attention_map: Attention map with shape (B, H, W, HW)
          Return:
             weighted V
        """

        B, C, H, W = V.shape
        V = V.permute(0, 2, 3, 1).reshape(B, H * W, C)  # B, HW, C

        weighted_V = torch.matmul(attention_map, V)

        weighted_V = weighted_V.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return weighted_V

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        x = self.layer_norm(x).permute(0, 3, 1, 2)  # B,C,H,W

        # Q branch
        Q = self.conv1x1_q(x)
        Q = self.deconv3x3_q(Q)

        # K branch
        K = self.conv1x1_k(x)
        K = self.deconv3x3_k(K)

        # V branch
        V = self.conv1x1_v(x)
        V = self.deconv3x3_v(V)

        # Channel Shuffle
        x = self.channel_shuffle(x)

        # DSA
        attention_map = self.depthwise_spatial_attention(Q, K)

        # SSA
        out = self.spatial_wise_self_attention(V, attention_map)

        # Apply weights
        out = F.relu(out)

        # 1x1 Conv out
        out = self.conv1x1_out(out)

        # Add and output
        out = out + self.stacked_conv(x)
        out = out + residual

        out = out.permute(0, 2, 3, 1)
        out = self.layer_norm2(out).permute(0, 3, 1, 2)

        return out


if __name__ == '__main__':
    # Example usage
    in_channels = 128  # Example number of input channels
    groups = 8  # Example number of groups for channel shuffle
    batch_size = 4
    height = 64
    width = 64

    # create a dummy input
    input_tensor = torch.randn(batch_size, in_channels, height, width)

    # create the AFIA module
    afia_module = AFIA(in_channels, groups)

    # forward propagation
    output_tensor = afia_module(input_tensor)

    # check the output shape
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)