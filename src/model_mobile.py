import torch
import torchvision
import torch.nn as nn

from torchsummary import summary


class ExtraBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        intermediate_channels = out_channels // 2
        self.act = nn.ReLU6()
        #1x1 projection
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        
        #3x3 depthwise 
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, groups=intermediate_channels)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        #1x1 projection to output_channels
        self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        return x
    

class PredictionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        #3x3 depthwise
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        #1x1 projection to output_channels
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.act = nn.ReLU6()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        return self.conv2(x)


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        # self.n_classes = n_classes

        self.base_model = torchvision.models.mobilenet_v3_large(torchvision.models.MobileNet_V3_Large_Weights.DEFAULT).features
        # self.conv1 = nn.Conv2d(960, 128, kernel_size=2, stride=1)
        # self.conv2 = nn.Conv2d(128, 64, kernel_size=3)
        # self.conv3 = nn.Conv2d(64, 30, kernel_size=1)
        self.extra1 = ExtraBlock(960, 512)
        self.extra2 = ExtraBlock(512, 256)
        self.extra3 = ExtraBlock(256, 128)
        self.prediction = PredictionBlock(128, 30)

    def forward(self, x):
        x = self.base_model(x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # print(x.shape)
        x = self.extra1(x)
        x = self.extra2(x)
        x = self.extra3(x)
        x = self.prediction(x)
        # x = x.reshape(-1, 6, x.shape[2], x.shape[3])
        # print(x.shape)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def get_params(self):
        total_params = 0
        for param in self.base_model.parameters():
            total_params += param.numel()
        # for param in self.conv1.parameters():
        #     total_params += param.numel()
        # for param in self.conv2.parameters():
        #     total_params += param.numel()
        # for param in self.conv3.parameters():
        #     total_params += param.numel()
        for param in self.extra1.parameters():
            total_params += param.numel()
        for param in self.extra2.parameters():
            total_params += param.numel()
        for param in self.extra3.parameters():
            total_params += param.numel()
        for param in self.prediction.parameters():
            total_params += param.numel()

        return total_params


if __name__ == "__main__":
    model = Detector().eval()
    input = torch.randn(16, 3, 480, 480)

    with torch.no_grad():
        out = model(input)

    print(out.shape)
    print(model.get_params())

    # summary(model, (3, 320, 320), batch_size=1, device='cpu')
