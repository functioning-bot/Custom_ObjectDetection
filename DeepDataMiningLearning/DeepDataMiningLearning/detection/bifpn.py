"""
Bidirectional Feature Pyramid Network (BiFPN)
Implementation based on EfficientDet paper: https://arxiv.org/abs/1911.09070
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from collections import OrderedDict

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class BiFPNBlock(nn.Module):
    """
    Single BiFPN block with weighted feature fusion
    
    Features flow:
    P3 -----> P3_td -----> P3_out
       ↘              ↗
    P4 -----> P4_td -----> P4_out
       ↘    ↗    ↘    ↗
    P5 -----> P5_td -----> P5_out
       ↘    ↗         ↗
    P6 -----> P6_td -----> P6_out
       ↘              ↗
    P7 ----------------> P7_out
    """
    
    def __init__(self, num_channels, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        
        # Top-down pathway convolutions
        self.conv6_td = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv5_td = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv4_td = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv3_td = DepthwiseSeparableConv(num_channels, num_channels)
        
        # Bottom-up pathway convolutions
        self.conv4_out = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv5_out = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv6_out = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv7_out = DepthwiseSeparableConv(num_channels, num_channels)
        
        # Learnable weights for feature fusion (fast normalized fusion)
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w1_relu = nn.ReLU()
        
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w2_relu = nn.ReLU()
    
    def forward(self, inputs):
        """
        Args:
            inputs: Dict of features {level: feature_map}
                   e.g., {'0': P3, '1': P4, '2': P5, '3': P6, 'pool': P7}
        """
        # Convert dict to list for easier indexing
        if isinstance(inputs, dict):
            p3, p4, p5, p6, p7 = inputs['0'], inputs['1'], inputs['2'], inputs['3'], inputs['pool']
        else:
            p3, p4, p5, p6, p7 = inputs
        
        # Top-down pathway
        # P6_td = Conv(P6 + Resize(P7))
        p7_up = F.interpolate(p7, size=p6.shape[-2:], mode='nearest')
        w1 = self.w1_relu(self.w1)
        w1_sum = w1.sum() + self.epsilon
        p6_td = self.conv6_td((w1[0] * p6 + w1[1] * p7_up) / w1_sum)
        
        # P5_td = Conv(P5 + Resize(P6_td))
        p6_td_up = F.interpolate(p6_td, size=p5.shape[-2:], mode='nearest')
        w1 = self.w1_relu(self.w1)
        w1_sum = w1.sum() + self.epsilon
        p5_td = self.conv5_td((w1[0] * p5 + w1[1] * p6_td_up) / w1_sum)
        
        # P4_td = Conv(P4 + Resize(P5_td))
        p5_td_up = F.interpolate(p5_td, size=p4.shape[-2:], mode='nearest')
        w1 = self.w1_relu(self.w1)
        w1_sum = w1.sum() + self.epsilon
        p4_td = self.conv4_td((w1[0] * p4 + w1[1] * p5_td_up) / w1_sum)
        
        # P3_out = Conv(P3 + Resize(P4_td))
        p4_td_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        w1 = self.w1_relu(self.w1)
        w1_sum = w1.sum() + self.epsilon
        p3_out = self.conv3_td((w1[0] * p3 + w1[1] * p4_td_up) / w1_sum)
        
        # Bottom-up pathway
        # P4_out = Conv(P4 + P4_td + Resize(P3_out))
        p3_out_down = F.max_pool2d(p3_out, kernel_size=2, stride=2)
        w2 = self.w2_relu(self.w2)
        w2_sum = w2.sum() + self.epsilon
        p4_out = self.conv4_out((w2[0] * p4 + w2[1] * p4_td + w2[2] * p3_out_down) / w2_sum)
        
        # P5_out = Conv(P5 + P5_td + Resize(P4_out))
        p4_out_down = F.max_pool2d(p4_out, kernel_size=2, stride=2)
        w2 = self.w2_relu(self.w2)
        w2_sum = w2.sum() + self.epsilon
        p5_out = self.conv5_out((w2[0] * p5 + w2[1] * p5_td + w2[2] * p4_out_down) / w2_sum)
        
        # P6_out = Conv(P6 + P6_td + Resize(P5_out))
        p5_out_down = F.max_pool2d(p5_out, kernel_size=2, stride=2)
        w2 = self.w2_relu(self.w2)
        w2_sum = w2.sum() + self.epsilon
        p6_out = self.conv6_out((w2[0] * p6 + w2[1] * p6_td + w2[2] * p5_out_down) / w2_sum)
        
        # P7_out = Conv(P7 + Resize(P6_out))
        p6_out_down = F.max_pool2d(p6_out, kernel_size=2, stride=2)
        w1 = self.w1_relu(self.w1)
        w1_sum = w1.sum() + self.epsilon
        p7_out = self.conv7_out((w1[0] * p7 + w1[1] * p6_out_down) / w1_sum)
        
        # Return as OrderedDict to match FPN output format
        return OrderedDict([
            ('0', p3_out),
            ('1', p4_out),
            ('2', p5_out),
            ('3', p6_out),
            ('pool', p7_out)
        ])

class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network
    
    Args:
        in_channels_list: List of input channels for each level [C3, C4, C5]
        out_channels: Number of output channels (same for all levels)
        num_layers: Number of BiFPN layers (default: 3)
    """
    
    def __init__(self, in_channels_list, out_channels, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # Lateral convolutions to convert inputs to uniform channel size
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        # Extra levels (P6, P7) using downsampling
        self.p6_conv = nn.Conv2d(in_channels_list[-1], out_channels, kernel_size=3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        # BiFPN blocks
        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(out_channels) for _ in range(num_layers)
        ])
        
        self.out_channels = out_channels
    
    def forward(self, x):
        """
        Args:
            x: OrderedDict of input features from backbone
               e.g., {'0': C3, '1': C4, '2': C5, '3': C6}
        
        Returns:
            OrderedDict of output features
            {'0': P3, '1': P4, '2': P5, '3': P6, 'pool': P7}
        """
        # Convert inputs to list
        features = list(x.values())
        
        # Apply lateral convolutions (C3, C4, C5 -> uniform channels)
        laterals = [
            conv(features[i]) for i, conv in enumerate(self.lateral_convs)
        ]
        
        # Add P6 and P7 through downsampling
        # P6 is obtained via a 3x3 stride-2 conv on C5
        p6 = self.p6_conv(features[-1])
        # P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6
        p7 = self.p7_conv(F.relu(p6))
        
        # Initial pyramid: [P3, P4, P5, P6, P7]
        pyramid = OrderedDict([
            ('0', laterals[0]),  # P3
            ('1', laterals[1]),  # P4
            ('2', laterals[2]),  # P5
            ('3', p6),           # P6
            ('pool', p7)         # P7
        ])
        
        # Apply BiFPN blocks
        for bifpn_block in self.bifpn_blocks:
            pyramid = bifpn_block(pyramid)
        
        return pyramid

# Simple test
if __name__ == "__main__":
    # Test BiFPN
    in_channels_list = [256, 512, 1024, 2048]  # ResNet50 channels
    out_channels = 256
    
    bifpn = BiFPN(in_channels_list, out_channels, num_layers=3)
    
    # Create dummy input
    x = OrderedDict([
        ('0', torch.rand(2, 256, 200, 200)),
        ('1', torch.rand(2, 512, 100, 100)),
        ('2', torch.rand(2, 1024, 50, 50)),
        ('3', torch.rand(2, 2048, 25, 25))
    ])
    
    out = bifpn(x)
    print("BiFPN output shapes:")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
