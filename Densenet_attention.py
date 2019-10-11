import torch
import torch.nn.functional as F
from torch import nn
from densenet import Densenet


class AADFNet(nn.Module):

    def __init__(self):
        super(AADFNet, self).__init__()
        densenet = Densenet()
        self.layer0 = densenet.layer0
        self.layer1 = densenet.layer1
        self.layer2 = densenet.layer2
        self.layer3 = densenet.layer3
        self.layer4 = densenet.layer4

        self.aspp_layer4 = _ASPP_attention(2208, 32)
        self.aspp_layer3 = _ASPP_attention(2112, 32)
        self.aspp_layer2 = _ASPP_attention(768, 32)
        self.aspp_layer1 = _ASPP_attention(384, 32)

        self.predict4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.predict41 = nn.Conv2d(32, 1, kernel_size=1)
        self.predict42 = nn.Conv2d(32, 1, kernel_size=1)
        self.predict43 = nn.Conv2d(32, 1, kernel_size=1)
        self.predict44 = nn.Conv2d(32, 1, kernel_size=1)


        self.predict3 = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.predict31 = nn.Conv2d(33, 1, kernel_size=1)
        self.predict32 = nn.Conv2d(33, 1, kernel_size=1)
        self.predict33 = nn.Conv2d(33, 1, kernel_size=1)
        self.predict34 = nn.Conv2d(33, 1, kernel_size=1)

        self.predict2 = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.predict21 = nn.Conv2d(33, 1, kernel_size=1)
        self.predict22 = nn.Conv2d(33, 1, kernel_size=1)
        self.predict23 = nn.Conv2d(33, 1, kernel_size=1)
        self.predict24 = nn.Conv2d(33, 1, kernel_size=1)

        self.predict1 = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.predict11 = nn.Conv2d(33, 1, kernel_size=1)
        self.predict12 = nn.Conv2d(33, 1, kernel_size=1)
        self.predict13 = nn.Conv2d(33, 1, kernel_size=1)
        self.predict14 = nn.Conv2d(33, 1, kernel_size=1)

        self.predict4_2 = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.predict3_2 = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.predict2_2 = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.residual3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.residual2_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.residual3_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.residual4_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        # layer3 = self.nlb(layer3)
        layer4 = self.layer4(layer3)

        aspp_layer41, aspp_layer42, aspp_layer43, aspp_layer44, aspp_layer4 = self.aspp_layer4(layer4)
        aspp_layer31, aspp_layer32, aspp_layer33, aspp_layer34, aspp_layer3 = self.aspp_layer3(layer3)
        aspp_layer21, aspp_layer22, aspp_layer23, aspp_layer24, aspp_layer2 = self.aspp_layer2(layer2)
        aspp_layer11, aspp_layer12, aspp_layer13, aspp_layer14, aspp_layer1 = self.aspp_layer1(layer1)


        predict41 = self.predict41(aspp_layer41)
        predict42 = self.predict42(aspp_layer42)
        predict43 = self.predict43(aspp_layer43)
        predict44 = self.predict44(aspp_layer44)

        predict4 = self.predict4(aspp_layer4)
        predict4 = F.upsample(predict4, size=layer3.size()[2:], mode='bilinear')
        aspp_layer4 = F.upsample(aspp_layer4, size=layer3.size()[2:], mode='bilinear')

        predict31 = self.predict31(torch.cat((predict4, aspp_layer31), 1)) + predict4
        predict32 = self.predict32(torch.cat((predict4, aspp_layer32), 1)) + predict4
        predict33 = self.predict33(torch.cat((predict4, aspp_layer33), 1)) + predict4
        predict34 = self.predict34(torch.cat((predict4, aspp_layer34), 1)) + predict4

        fpn_layer3 = aspp_layer3 + self.residual3(torch.cat((aspp_layer4, aspp_layer3), 1))
        predict3 = self.predict3(torch.cat((predict4, fpn_layer3), 1)) + predict4
        predict3 = F.upsample(predict3, size=layer2.size()[2:], mode='bilinear')
        fpn_layer3 = F.upsample(fpn_layer3, size=layer2.size()[2:], mode='bilinear')

        predict21 = self.predict21(torch.cat((predict3, aspp_layer21), 1)) + predict3
        predict22 = self.predict22(torch.cat((predict3, aspp_layer22), 1)) + predict3
        predict23 = self.predict23(torch.cat((predict3, aspp_layer23), 1)) + predict3
        predict24 = self.predict24(torch.cat((predict3, aspp_layer24), 1)) + predict3

        fpn_layer2 = aspp_layer2 + self.residual2(torch.cat((fpn_layer3, aspp_layer2), 1))
        predict2 = self.predict2(torch.cat((predict3, fpn_layer2), 1)) + predict3
        predict2 = F.upsample(predict2, size=layer1.size()[2:], mode='bilinear')
        fpn_layer2 = F.upsample(fpn_layer2, size=layer1.size()[2:], mode='bilinear')

        predict11 = self.predict11(torch.cat((predict2, aspp_layer11), 1)) + predict2
        predict12 = self.predict12(torch.cat((predict2, aspp_layer12), 1)) + predict2
        predict13 = self.predict13(torch.cat((predict2, aspp_layer13), 1)) + predict2
        predict14 = self.predict14(torch.cat((predict2, aspp_layer14), 1)) + predict2

        fpn_layer1 = aspp_layer1 + self.residual1(torch.cat((fpn_layer2, aspp_layer1), 1))
        predict1 = self.predict1(torch.cat((predict2, fpn_layer1), 1)) + predict2

        fpn_layer4 = F.upsample(aspp_layer4, size=layer1.size()[2:], mode='bilinear')
        fpn_layer3 = F.upsample(fpn_layer3, size=layer1.size()[2:], mode='bilinear')
        fpn_layer2 = F.upsample(fpn_layer2, size=layer1.size()[2:], mode='bilinear')

        fpn_layer2_2 = fpn_layer2 + self.residual2_2(torch.cat((fpn_layer2, fpn_layer1), 1))
        predict2_2 = self.predict2_2(torch.cat((predict1, fpn_layer2_2), 1)) + predict1

        fpn_layer3_2 = fpn_layer3 + self.residual3_2(torch.cat((fpn_layer3, fpn_layer2), 1))
        predict3_2 = self.predict3_2(torch.cat((predict2_2, fpn_layer3_2), 1)) + predict2_2

        fpn_layer4_2 = fpn_layer4 + self.residual4_2(torch.cat((fpn_layer4, fpn_layer3), 1))
        predict4_2 = self.predict4_2(torch.cat((predict3_2, fpn_layer4_2), 1)) + predict3_2

        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')

        predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')
        predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')

        predict44 = F.upsample(predict44, size=x.size()[2:], mode='bilinear')
        predict43 = F.upsample(predict43, size=x.size()[2:], mode='bilinear')
        predict42 = F.upsample(predict42, size=x.size()[2:], mode='bilinear')
        predict41 = F.upsample(predict41, size=x.size()[2:], mode='bilinear')

        predict34 = F.upsample(predict34, size=x.size()[2:], mode='bilinear')
        predict33 = F.upsample(predict33, size=x.size()[2:], mode='bilinear')
        predict32 = F.upsample(predict32, size=x.size()[2:], mode='bilinear')
        predict31 = F.upsample(predict31, size=x.size()[2:], mode='bilinear')

        predict24 = F.upsample(predict24, size=x.size()[2:], mode='bilinear')
        predict23 = F.upsample(predict23, size=x.size()[2:], mode='bilinear')
        predict22 = F.upsample(predict22, size=x.size()[2:], mode='bilinear')
        predict21 = F.upsample(predict21, size=x.size()[2:], mode='bilinear')

        predict14 = F.upsample(predict14, size=x.size()[2:], mode='bilinear')
        predict13 = F.upsample(predict13, size=x.size()[2:], mode='bilinear')
        predict12 = F.upsample(predict12, size=x.size()[2:], mode='bilinear')
        predict11 = F.upsample(predict11, size=x.size()[2:], mode='bilinear')

        if self.training:
            return predict4_2, predict3_2, predict2_2, predict1, predict2, predict3, predict4,\
                   predict41, predict42, predict43, predict44, \
                   predict31, predict32, predict33, predict34, \
                   predict21, predict22, predict23, predict24, \
                   predict11, predict12, predict13, predict14,
        return F.sigmoid(predict4_2)


class _ASPP_attention(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(_ASPP_attention, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim + out_dim * 2, out_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim + out_dim * 3, out_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(out_dim), nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )

        self.attention4_local = nn.Sequential(
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.Softmax2d()
        )
        self.attention3_local = nn.Sequential(
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.Softmax2d()
        )
        self.attention2_local = nn.Sequential(
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.Softmax2d()
        )
        self.attention1_local = nn.Sequential(
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.Softmax2d()
        )

        self.attention4_global = nn.Sequential(
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.Softmax2d()
        )
        self.attention3_global = nn.Sequential(
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.Softmax2d()
        )
        self.attention2_global = nn.Sequential(
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.Softmax2d()
        )
        self.attention1_global = nn.Sequential(
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.Softmax2d()
        )


        self.refine4 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )



    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), 1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), 1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), 1))

        fusion4_123 = self.fuse4(torch.cat((conv1, conv2, conv3), 1))
        fusion3_124 = self.fuse3(torch.cat((conv1, conv2, conv4), 1))
        fusion2_134 = self.fuse2(torch.cat((conv1, conv3, conv4), 1))
        fusion1_234 = self.fuse1(torch.cat((conv2, conv3, conv4), 1))

        attention4_local = self.attention4_local(torch.cat((conv4, fusion4_123), 1))
        attention3_local = self.attention3_local(torch.cat((conv3, fusion3_124), 1))
        attention2_local = self.attention2_local(torch.cat((conv2, fusion2_134), 1))
        attention1_local = self.attention1_local(torch.cat((conv1, fusion1_234), 1))

        attention4_global = F.upsample(self.attention4_global(F.adaptive_avg_pool2d(torch.cat((conv4, fusion4_123), 1), 1)),
                                       size=x.size()[2:], mode='bilinear', align_corners=True)
        attention3_global = F.upsample(self.attention3_global(F.adaptive_avg_pool2d(torch.cat((conv3, fusion3_124), 1), 1)),
                                       size=x.size()[2:], mode='bilinear', align_corners=True)
        attention2_global = F.upsample(self.attention2_global(F.adaptive_avg_pool2d(torch.cat((conv2, fusion2_134), 1), 1)),
                                       size=x.size()[2:], mode='bilinear', align_corners=True)
        attention1_global = F.upsample(self.attention1_global(F.adaptive_avg_pool2d(torch.cat((conv1, fusion1_234), 1), 1)),
                                       size=x.size()[2:], mode='bilinear', align_corners=True)

        refine4 = self.refine4(torch.cat((fusion4_123 * attention4_local, fusion4_123 * attention4_global, conv4), 1))
        refine3 = self.refine3(torch.cat((fusion3_124 * attention3_local, fusion3_124 * attention3_global, conv3), 1))
        refine2 = self.refine2(torch.cat((fusion2_134 * attention2_local, fusion2_134 * attention2_global, conv2), 1))
        refine1 = self.refine1(torch.cat((fusion1_234 * attention1_local, fusion1_234 * attention1_global, conv1), 1))
        refine_fusion = torch.cat((refine1, refine2, refine3, refine4), 1)

        return refine1, refine2, refine3, refine4, refine_fusion
