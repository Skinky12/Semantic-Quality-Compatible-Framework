

import torch as torch
import torch.nn as nn
import scipy.io as scio
from .resnet_backbone import resnet50_backbone
from .transenc import FeatureAggretation

from .UpScale import UpBlock


class SmRmSepNet(nn.Module):
    def __init__(self):
        super(SmRmSepNet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.qdense = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.upl2tol1 = UpBlock('./New_up2to1.pth', [512, 256])
        self.upl3tol2 = UpBlock('./New_up3to2.pth', [1024, 512])
        self.upl4tol3 = UpBlock('./New_up4to3.pth', [2048, 1024])

        self.squeeze1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 8 - 1, stride=8, padding=8 - 1),
        )
        self.squeeze2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 4 - 1, stride=4, padding=4 - 1),
        )
        self.squeeze3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 2 - 1, stride=2, padding=2 - 1),
        )

        self.SmQlMerge = nn.Sequential(
            nn.Conv1d(in_channels=384, out_channels=192, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=192, out_channels=96, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=96, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=192, kernel_size=1),
            # nn.ReLU(),
        )

        # 6->48->24->1
        self.ScaleMerge = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=24, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=24, out_channels=24, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=24, out_channels=1, kernel_size=1),
        )

    def forward(self, ft):
        sf1 = ft['f1'].detach()
        sf2 = ft['f2'].detach()
        sf3 = ft['f3'].detach()
        sf4 = ft['f4'].detach()
        rf3 = self.upl4tol3(sf4.detach())
        rf2 = self.upl3tol2(sf3.detach())
        rf1 = self.upl2tol1(sf2.detach())

        qdiff1 = torch.abs(rf1.detach() - sf1.detach())
        qdiff2 = torch.abs(rf2.detach() - sf2.detach())
        qdiff3 = torch.abs(rf3.detach() - sf3.detach())

        qdiff1_r, qdiff2_r, qdiff3_r = qdiff1.view(qdiff1.size(0), qdiff1.size(1), -1), qdiff2.view(qdiff2.size(0),
                                                                                                    qdiff2.size(1),
                                                                                                    -1), qdiff3.view(
            qdiff3.size(0), qdiff3.size(1), -1)
        sf1_r, sf2_r, sf3_r = sf1.view(sf1.size(0), sf1.size(1), -1), sf2.view(sf2.size(0), sf2.size(1), -1), sf3.view(
            sf3.size(0), sf3.size(1), -1)
        scale1_sim = (torch.sum(qdiff1_r * sf1_r, keepdim=True, dim=-1) + 1e-5) / ((torch.sqrt(
            torch.sum(qdiff1_r * qdiff1_r, keepdim=True, dim=-1)) * torch.sqrt(
            torch.sum(sf1_r * sf1_r, keepdim=True, dim=-1))) + 1e-5)
        scale2_sim = (torch.sum(qdiff2_r * sf2_r, keepdim=True, dim=-1) + 1e-5) / ((torch.sqrt(
            torch.sum(qdiff2_r * qdiff2_r, keepdim=True, dim=-1)) * torch.sqrt(
            torch.sum(sf2_r * sf2_r, keepdim=True, dim=-1))) + 1e-5)
        scale3_sim = (torch.sum(qdiff3_r * sf3_r, keepdim=True, dim=-1) + 1e-5) / ((torch.sqrt(
            torch.sum(qdiff3_r * qdiff3_r, keepdim=True, dim=-1)) * torch.sqrt(
            torch.sum(sf3_r * sf3_r, keepdim=True, dim=-1))) + 1e-5)

        scale1_sim = (1 - torch.unsqueeze(torch.abs(scale1_sim), dim=-1))
        scale2_sim = (1 - torch.unsqueeze(torch.abs(scale2_sim), dim=-1))
        scale3_sim = (1 - torch.unsqueeze(torch.abs(scale3_sim), dim=-1))

        qdiff1, f1 = self.squeeze1(qdiff1 * scale1_sim.detach()), self.squeeze1(sf1 * (1 - scale1_sim.detach()))
        qdiff2, f2 = self.squeeze2(qdiff2 * scale2_sim.detach()), self.squeeze2(sf2 * (1 - scale2_sim.detach()))
        qdiff3, f3 = self.squeeze3(qdiff3 * scale3_sim.detach()), self.squeeze3(sf3 * (1 - scale3_sim.detach()))

        b, c, h, w = f1.size()
        qdiff1 = qdiff1.permute([0, 2, 3, 1])
        ft_q1 = qdiff1.view(b, h*w, c)
        f1 = f1.permute([0, 2, 3, 1])
        ft_s1 = f1.view(b, h*w, c)
        qft1 = torch.cat((ft_q1, ft_s1), dim=1)
        qft1 = self.SmQlMerge(qft1)
        qft1 = qft1.view(b, h, w, c)
        qft1 = qft1.permute([0, 3, 1, 2])

        b, c, h, w = f2.size()
        qdiff2 = qdiff2.permute([0, 2, 3, 1])
        ft_q2 = qdiff2.view(b, h*w, c)
        f2 = f2.permute([0, 2, 3, 1])
        ft_s2 = f2.view(b, h*w, c)
        qft2 = torch.cat((ft_q2, ft_s2), dim=1)
        qft2 = self.SmQlMerge(qft2)
        qft2 = qft2.view(b, h, w, c)
        qft2 = qft2.permute([0, 3, 1, 2])

        b, c, h, w = f3.size()
        qdiff3 = qdiff3.permute([0, 2, 3, 1])
        ft_q3 = qdiff3.view(b, h * w, c)
        f3 = f3.permute([0, 2, 3, 1])
        ft_s3 = f3.view(b, h * w, c)
        qft3 = torch.cat((ft_q3, ft_s3), dim=1)
        qft3 = self.SmQlMerge(qft3)
        qft3 = qft3.view(b, h, w, c)
        qft3 = qft3.permute([0, 3, 1, 2])

        qft1_a = self.avg_pool(qft1).view(qft1.size(0), -1)
        qft2_a = self.avg_pool(qft2).view(qft2.size(0), -1)
        qft3_a = self.avg_pool(qft3).view(qft3.size(0), -1)

        qft1_b = self.max_pool(qft1).view(qft1.size(0), -1)
        qft2_b = self.max_pool(qft2).view(qft2.size(0), -1)
        qft3_b = self.max_pool(qft3).view(qft3.size(0), -1)

        qft1 = torch.cat((qft1_a, qft1_b), dim=1)
        qft2 = torch.cat((qft2_a, qft2_b), dim=1)
        qft3 = torch.cat((qft3_a, qft3_b), dim=1)

        qft_merge = torch.cat((torch.unsqueeze(qft1, dim=1), torch.unsqueeze(qft2, dim=1),
                               torch.unsqueeze(qft3, dim=1)), dim=1)
        qft_pool = torch.squeeze(self.ScaleMerge(qft_merge))
        qsc = self.qdense(qft_pool)

        out = {}
        out['f1'] = sf1
        out['f2'] = sf2
        out['f3'] = sf3
        out['r3'] = rf3
        out['r2'] = rf2
        out['r1'] = rf1

        out['df3'] = qft3
        out['df2'] = qft2
        out['df1'] = qft1
        out['Q'] = qsc
        return out

