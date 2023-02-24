from torchvision.models import resnet50
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F


def unified_net():
    u_net = resnet50(pretrained=False)
    u_net.conv1 = nn.Identity()
    u_net.bn1 = nn.Identity()
    u_net.relu = nn.Identity()
    u_net.maxpool = nn.Identity()
    u_net.layer1 = nn.Identity()
    u_net.layer2 = nn.Identity()
    return u_net


class MultiScaleNet(nn.Module):
    def __init__(self):
        super(MultiScaleNet, self).__init__()
        self.large_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),resnet50(pretrained=False).layer1,resnet50(pretrained=False).layer2
        )
        self.mid_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),resnet50(pretrained=False).layer1,resnet50(pretrained=False).layer2
        )
        self.small_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),resnet50(pretrained=False).layer1,resnet50(pretrained=False).layer2
        )
        self.unified_net = unified_net()
        self.small_size = (32, 32)
        self.mid_size = (128, 128)
        self.large_size = (224, 224)
        self.unified_size = (28, 28)

    def forward(self, imgs):
        small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')

        z1 = self.small_net(small_imgs)
        z2 = self.mid_net(mid_imgs)
        z3 = self.large_net(large_imgs)

        z1 = F.interpolate(z1, size=self.unified_size, mode='bilinear')
        z2 = F.interpolate(z2, size=self.unified_size, mode='bilinear')

        y1 = self.unified_net(z1)
        y2 = self.unified_net(z2)
        y3 = self.unified_net(z3)

        return z1, z2, z3, y1, y2, y3


class MSNetPL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.multi_scale_net = MultiScaleNet()

    def training_step(self, batch, batch_idx):
        x, y = batch
        z1, z2, z3, y1, y2, y3 = self.multi_scale_net(x)

        si_loss1 = F.mse_loss(z1, z2)
        si_loss2 = F.mse_loss(z1, z3)
        si_loss3 = F.mse_loss(z2, z3)
        ce_loss1 = F.cross_entropy(y1, y)
        ce_loss2 = F.cross_entropy(y2, y)
        ce_loss3 = F.cross_entropy(y3, y)
        total_loss = si_loss1 + si_loss2 + si_loss3 + ce_loss1 + ce_loss2 + ce_loss3
        loss_dict = {
            "si_loss1": si_loss1,
            "si_loss2": si_loss2,
            "si_loss3": si_loss3,
            "ce_loss1": ce_loss1,
            "ce_loss2": ce_loss2,
            "ce_loss3": ce_loss3,
            "total_loss": total_loss}
        self.log_dict(loss_dict)

        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        pass


if __name__ == '__main__':
    # small_imgs = torch.rand(8, 3, 32, 32)
    # mid_imgs = torch.rand(8, 3, 128, 128)
    large_imgs = torch.rand(8, 3, 224, 224)
    m_model = MultiScaleNet()
    out = m_model(large_imgs)
    print("end")
