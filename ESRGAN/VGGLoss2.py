import torch.nn as nn
from torchvision.models import vgg19
import torch
import torch.nn.functional as F
# phi_5,4 5th conv layer before maxpooling but after activation

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        print('===> Loading VGG model')
        netVGG = vgg19()
        netVGG.load_state_dict(torch.load('./VGG19/vgg19-dcbb9e9d.pth'))
        self.vgg = netVGG.features[:35].eval().to(device)  # VGG54, before activation
        self.loss = nn.L1Loss()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # # The preprocessing method of the input data. This is the preprocessing method of the VGG model on the ImageNet dataset.
        # self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        # self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, input, target):
        # input = (input - self.mean) / self.std
        # target = (target - self.mean) / self.std
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    v = VGGLoss(device)
