# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from torchvision.models import resnet18
from torchvision.models import resnet50
import torch
import torch.nn as nn


def unified_net():
    u_net = resnet50(pretrained=True)
    u_net.conv1 = nn.Identity()
    u_net.bn1 = nn.Identity()
    u_net.relu = nn.Identity()
    u_net.maxpool = nn.Identity()
    return u_net


class MultiScaleNet(nn.Module):
    def __init__(self):
        super(MultiScaleNet, self).__init__()
        self.large_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.mid_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.small_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.unified_net = unified_net()

    def forward(self, multi_input):
        [small_imgs, mid_imgs, large_imgs] = multi_input
        z1 = self.small_net(small_imgs)
        z2 = self.mid_net(mid_imgs)
        z3 = self.large_net(large_imgs)
        y1 = self.unified_net(z1)
        y2 = self.unified_net(z2)
        y3 = self.unified_net(z3)
        return z1, z2, z3, y1, y2, y3

    def small_forward(self, small_imgs):
        x = self.small_net(small_imgs)
        x = self.unified_net(x)
        return x

    def mid_forward(self, mid_imgs):
        x = self.mid_net(mid_imgs)
        x = self.unified_net(x)
        return x

    def large_forward(self, large_imgs):
        x = self.large_net(large_imgs)
        x = self.unified_net(x)
        return x


if __name__ == '__main__':
    small_imgs = torch.rand(8, 3, 32, 32)
    mid_imgs = torch.rand(8, 3, 128, 128)
    large_imgs = torch.rand(8, 3, 224, 224)
    m_model = MultiScaleNet()
    input = [small_imgs, mid_imgs, large_imgs]
    out = m_model(input)
    print("end")