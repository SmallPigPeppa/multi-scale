from torchvision.models import resnet50
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F



class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.resnet=resnet50(pretrained=False)
        self.small_size = (32, 32)
        self.mid_size = (128, 128)
        self.large_size = (224, 224)


    def forward(self, imgs):
        small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')

        small_imgs = F.interpolate(small_imgs, size=self.large_size, mode='bilinear')
        mid_imgs = F.interpolate(mid_imgs, size=self.large_size, mode='bilinear')



        y1 = self.resnet(small_imgs)
        y2 = self.resnet(mid_imgs)
        y3 = self.resnet(large_imgs)


        return y1, y2, y3



if __name__ == '__main__':
    # small_imgs = torch.rand(8, 3, 32, 32)
    # mid_imgs = torch.rand(8, 3, 128, 128)
    large_imgs = torch.rand(8, 3, 224, 224)
    m_model = BaselineNet()
    out = m_model(large_imgs)
    print("end")



