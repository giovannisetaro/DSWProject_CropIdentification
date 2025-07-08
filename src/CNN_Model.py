import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalPixelCNNV1(nn.Module):
    def __init__(self, in_channels=4, out_channels=64, kernel_size=3):
        #in_channels : number of bands
        # out_channels : number of filter (hyper parametre)
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        # padding=kernel_size//2 so we keep the same temporal dimention, here (12 → 12).
        self.bn = nn.BatchNorm1d(out_channels)
        # Batch Normalization to stabilize the learning 
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C=4, T=12, H=24, W=24]
        B, C, T, H, W = x.shape

        # Reshape pour traiter chaque pixel indépendamment / pixel wise
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, C, T]

        #convolution 1D temporelle/multibande pixel-wise
        x = x.view(B*H*W, C, T)                    # [B*H*W, C, T]
        
        x = self.conv1d(x)        # [B*H*W, D=number of filters, T (bcz we choose a padding=kernel_size//2)]
        x = self.bn(x)
        x = self.relu(x)

        # Option 1: robust to noise but lost of peaks / rapid variations (harvest/ blooming)

       # x = x.mean(dim=2)    # [B*H*W, D]
        #Option 2 – Max temporel : sensitive to peaks / rapid variations but ignore the global dynamic 
        x = x.max(dim=2).values  # [B*H*W, D]

        #option 3 : attention module but .... no time

        # Reformer en tenseur image embedding
        D = x.size(1)
        x = x.view(B, H, W, D).permute(0, 3, 1, 2) # [B, D, H, W]
        return x
    
############## ADDING +1 conv layer to have a bigger receptive temporal field in the embedding ! 

class TemporalPixelCNN(nn.Module):
    def __init__(self, in_channels=4, out_channels=64, kernel_size=3):
        super().__init__()
        # First 1D convolution over the temporal dimension
        # Kernel size defines how many months are looked at simultaneously
        # Padding is set to keep the temporal dimension unchanged (same length output)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)  # BatchNorm to stabilize training
        self.relu = nn.ReLU()

        # Second 1D convolution to increase temporal receptive field
        # By stacking two conv layers, the model captures longer temporal patterns (e.g. >3 months)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x shape: [B, C=4, T=12, H=24, W=24]
        B, C, T, H, W = x.shape

        # Permute to bring spatial dims upfront, so that we can treat each pixel independently
        # New shape: [B, H, W, C, T]
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        # Flatten batch and spatial dims to process each pixel time series independently
        # Shape: [B*H*W, C, T]
        x = x.view(B*H*W, C, T)

        # First temporal convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Second temporal convolution expands the temporal receptive field
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Temporal pooling (max) reduces temporal dimension, keeps strongest temporal features
        # Result shape: [B*H*W, out_channels]
        x = x.max(dim=2).values

        # Reshape back to image embedding form for spatial CNN input
        # Shape: [B, out_channels, H, W]
        D = x.size(1)
        x = x.view(B, H, W, D).permute(0, 3, 1, 2)

        return x









class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        #in_ch : nombre de canaux d’entrée / nbr of features ie dimension of the multispectral + temporal embedding
        # out_ch : number of output channels 
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), #kernel depth = nbr of feature = D = initiali : number of filters in the temporal CNN 
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), # negativ values set to 0 for optimisation
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), ## Receptiv field of 5*5 pixel <=> 50m*50m
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch , in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ajuster taille si besoin
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x)      # [B,64,H,W]
        x2 = self.down1(x1)   # [B,128,H/2,W/2]
        x3 = self.down2(x2)   # [B,256,H/4,W/4]
        x = self.up1(x3, x2)  # [B,128,H/2,W/2]
        x = self.up2(x, x1)   # [B,64,H,W]
        logits = self.outc(x) # [B,out_ch,H,W]
        return logits
    



class CropTypeClassifier(nn.Module):
    def __init__(self, num_classes, temporal_out_channels=64):
        super().__init__()
        self.temporal_cnn = TemporalPixelCNN(in_channels=4, out_channels=temporal_out_channels)
        self.unet = UNet(in_ch=temporal_out_channels, out_ch=num_classes)

    def forward(self, x):
        x = self.temporal_cnn(x)   # [B, temporal_out_channels, 24, 24]
        x = self.unet(x)           # [B, num_classes, 24, 24]

        # x final shape : [B, 4, 12, 24, 24]
        return x
    

if __name__ == "__main__":
    B, C, T, H, W = 2, 4, 12, 24, 24
    num_classes = 25

    model = CropTypeClassifier(num_classes=num_classes)
    x = torch.randn(B, C, T, H, W)
    output = model(x)
    print(output.shape)  # doit afficher : torch.Size([2, 5, 24, 24])