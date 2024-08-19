import math
import torch
import torch
import torch.nn as nn   
import torch.nn.functional as F
        
class Basicblock(nn.Module):
    def __init__(self, num):
        super(Basicblock,self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(3, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU()
            )
    def forward(self, x):
        return self.m(x)

class Downblock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(Downblock, self).__init__()
        self.dwconv = nn.Conv2d(channels, channels, groups=channels, stride=2,
                                kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        return self.dwconv(x)

class ACMBlock(nn.Module):
        #aggregation-calibration mechanism
    def __init__(self, in_planes, out_planes, spatial=7, extent=0, extra_params=True, mlp=True, dropRate=0.3):
        super(ACMBlock, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.equalInOut = (in_planes == out_planes)

        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes,
            out_planes, kernel_size=1, stride=1, bias=False) or None

        if extra_params:
            if extent: modules = [Downblock(out_planes)]
            for i in range((extent-1) // 2): modules.append(nn.Sequential(nn.ReLU(inplace=True), Downblock(out_planes)))
            self.downop = nn.Sequential(*modules) if extent else Downblock(out_planes, kernel_size=spatial)
        else:
            self.downop = nn.AdaptiveAvgPool2d(spatial // extent) if extent else nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(nn.Conv2d(out_planes, out_planes // 16, kernel_size=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(out_planes // 16, out_planes, kernel_size=1, bias=False)) if mlp else lambda x: x

    def forward(self, x):
        out = self.conv(x)
        mp = self.mlp(self.downop(out))
        # Assuming squares because lazy.
        mp = F.interpolate(mp, out.shape[-1])
        if not self.equalInOut: x = self.convShortcut(x)
        #print(out.size())
        return torch.add(x, out * torch.sigmoid(mp))

class Colorbranchgeb(nn.Module):
    def __init__(self, num,):
        super(Colorbranchgeb,self).__init__()
        
        self.downd = nn.Sequential(
            ACMBlock(num, num),
            nn.ReLU(),
            nn.Conv2d(num, 3, 3, 1, 1),
            nn.Sigmoid()
            )
        self.upd = nn.Sequential(
            nn.Conv2d(3, num, 3, 1, 1),
            nn.ReLU(),
            ACMBlock(num, num),
            nn.ReLU()
            )
    def forward(self, x):
        c = self.downd(x)
        return self.upd(c)+x, c

class Detailbranchgeb(nn.Module):
    def __init__(self, num,rate=0.5):
        super(Detailbranchgeb,self).__init__()
        
        self.downd = nn.Sequential(
            ACMBlock(num, int(num*rate)),
            nn.ReLU(),
            nn.Conv2d(int(num*rate), 1, 3, 1, 1),
            nn.Sigmoid()
            )
        self.upd = nn.Sequential(
            nn.Conv2d(1, int(num*rate), 3, 1, 1),
            nn.ReLU(),
            ACMBlock(int(num*rate), num),
            nn.ReLU()
            )
    def forward(self, x):
        d = self.downd(x)
        return self.upd(d)+x, d
        
class DAM(nn.Module):
    def __init__(self, num):
        super(DAM,self).__init__()
        self.c = Colorbranchgeb(num)
        self.d = Detailbranchgeb(num)
        
    def forward(self, x):
        u1, c= self.c(x)
        u2, d= self.d(x)

        return u1+u2, d*c, d, c

class Head(nn.Module):
    def __init__(self, num):
        super(Head, self).__init__()
        self.r = nn.Sequential(
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, 3, 3, 1, 1),
            nn.Sigmoid()
            )
        self.l = nn.Sequential(
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, 1, 3, 1, 1),
            nn.Sigmoid() 
            )
    def forward(self, x):
        return self.r(x), self.l(x)
        
class CusLIE(nn.Module):  
    def __init__(self, num=64):
        super(CusLIE,self).__init__()
        self.fl = Basicblock(num)
        self.da = DAM(num)
        self.head = Head(num)
        
    def forward(self, x):
        x = self.fl(x)
        u,X,_,_ = self.da(x)
        R,L = self.head(u)
        return L, R, X      
