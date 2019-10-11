from torchvision import models
import torch
from torch import nn
from config import pretrained_densenet_path
import re


class Densenet(nn.Module):
    def __init__(self):
        super(Densenet, self).__init__()
        densenet = models.densenet161()

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_densenet_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        densenet.load_state_dict(state_dict)

        self.layer0 = nn.Sequential(densenet.features.conv0, densenet.features.norm0, densenet.features.relu0,densenet.features.pool0)
        self.layer1 = nn.Sequential(densenet.features.denseblock1)
        self.layer2 = nn.Sequential(densenet.features.transition1, densenet.features.denseblock2)
        self.layer3 = nn.Sequential(densenet.features.transition2, densenet.features.denseblock3)
        self.layer4 = nn.Sequential(densenet.features.transition3, densenet.features.denseblock4)


    def forward(self, x):
        layer0 = self.layer0(x)

        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


