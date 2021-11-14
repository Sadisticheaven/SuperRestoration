import torch.nn as nn
from torchvision.models import vgg19
import torch
# phi_5,4 5th conv layer before maxpooling but after activation

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        print('===> Loading VGG model')
        netVGG = vgg19()
        netVGG.load_state_dict(torch.load('./VGG19/vgg19-dcbb9e9d.pth'))
        self.vgg = netVGG.features[:35].eval().to(device)  # VGG54, before activation
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)