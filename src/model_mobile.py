import torch
import torchvision
import torch.nn as nn

from torchsummary import summary

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        # self.n_classes = n_classes

        self.base_model = torchvision.models.mobilenet_v3_large(torchvision.models.MobileNet_V3_Large_Weights.DEFAULT).features
        self.conv1 = nn.Conv2d(960, 128, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 30, kernel_size=1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = x.reshape(-1, 6, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def get_params(self):
        total_params = 0
        for param in self.base_model.parameters():
            total_params += param.numel()
        for param in self.conv1.parameters():
            total_params += param.numel()
        for param in self.conv2.parameters():
            total_params += param.numel()
        for param in self.conv3.parameters():
            total_params += param.numel()

        return total_params


if __name__ == "__main__":
    model = Detector().eval()
    input = torch.randn(16, 3, 320, 320)

    with torch.no_grad():
        out = model(input)

    print(out.shape)
    print(model.get_params())

    # summary(model, (3, 320, 320), batch_size=1, device='cpu')
