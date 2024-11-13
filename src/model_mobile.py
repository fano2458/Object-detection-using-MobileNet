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


class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, 128, 1) for in_ch in in_channels])
        self.top_down_convs = nn.ModuleList([nn.Conv2d(128, 128, 3, padding=1) for _ in range(len(in_channels) - 1)])
        self.convs = nn.ModuleList([nn.Conv2d(128, 128, 3) for _ in range(len(in_channels)-1)])

    def forward(self, x):
        outputs = [self.lateral_convs[0](x[0])]

        for i in range(1, len(x)):
            outputs.append(self.lateral_convs[i](x[i]))
            # print("###############")
            # print(outputs[-1].shape)
            # print(self.convs[i-1](outputs[-2]).shape)
            # print("###############")
            outputs[-1] = outputs[-1] + self.convs[i-1](outputs[-2]) # reduce size of previous for addition
            # outputs[-1] = outputs[-1] + nn.functional.interpolate(outputs[-2], size=outputs[-1].shape[-2:], mode='nearest')
            outputs[-1] = self.top_down_convs[i-1](outputs[-1])

        return outputs


class Detector(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = torchvision.models.mobilenet_v3_large(torchvision.models.MobileNet_V3_Large_Weights.DEFAULT).features
        self.extra1 = ExtraBlock(960, 512)
        self.extra2 = ExtraBlock(512, 256)
        self.extra3 = ExtraBlock(256, 128)
        self.prediction = PredictionBlock(128, 30)
        self.fpn = FPN([960, 512, 256, 128])

    def forward(self, x):
        x = self.base_model(x)
        features = [x]
        x = self.extra1(x)
        features.append(x)
        x = self.extra2(x)
        features.append(x)
        x = self.extra3(x)
        features.append(x)

        features = self.fpn(features)
        x = features[-1]
        
        x = self.prediction(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def get_params(self):
        # TODO rewrite this method
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()

        return total_params


if __name__ == "__main__":
    model = Detector().eval()
    input = torch.randn(16, 3, 480, 480)

    with torch.no_grad():
        out = model(input)

    print(out.shape)
    print(model.get_params())

    # summary(model, (3, 480, 480), batch_size=1, device='cpu')
