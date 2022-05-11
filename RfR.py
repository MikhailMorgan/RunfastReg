import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import RfR_utils

class RfR_model(nn.Module):

    #inter_in_channels = 64
    #last_in_channels = 32 + 3 # 32(context) + 3(flow)
    
    def __init__(self, name, in_channels, out_channels):
        super(RfR_model, self).__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.inputL = RfR_utils.DoubleConv(self.in_channels, self.in_channels)
        self.down1 = RfR_utils.Down(self.in_channels, 2*self.in_channels)
        self.down2 = RfR_utils.Down(2*self.in_channels, 2*self.in_channels)
        self.down3 = RfR_utils.Down(2*self.in_channels, 4*self.in_channels)
        self.down4 = RfR_utils.Down(4*self.in_channels, 4*self.in_channels)
        self.down5 = RfR_utils.Down(4*self.in_channels, 8*self.in_channels)
        self.down6 = RfR_utils.Down(8*self.in_channels, 8*self.in_channels)
        
        self.up1 = RfR_utils.Up(8*self.in_channels, 8*self.in_channels)
        self.up2 = RfR_utils.Up(8*self.in_channels, 4*self.in_channels)
        self.up3 = RfR_utils.Up(4*self.in_channels, 4*self.in_channels)
        self.up4 = RfR_utils.Up(4*self.in_channels, 2*self.in_channels)
        self.up5 = RfR_utils.Up(2*self.in_channels, 2*self.in_channels)
        self.up6 = RfR_utils.Up(2*self.in_channels, self.in_channels)
        self.outputL = RfR_utils.OutConv(self.in_channels, self.out_channels)
        
    def forward(self, x):
        x1 = self.inputL(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        b = self.down6(x6)
        
        x = self.up1(b, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        
        x = self.outputL(x)
        
        return x
        