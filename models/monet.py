from models.common import *
from models.monet_s_set import Set


class Predict32(Module):

    def __init__(self, f64_channels, f32channels, nc):
        super(Predict32, self).__init__()
        self.c1 = ConvolutionLayer(f64_channels, f32channels, kernel_size=3, stride=2)
        channels = f32channels + f32channels
        self.c2 = nn.Sequential(
            ConvolutionLayer(channels, f32channels, kernel_size=1, padding=0, action='leaky'),
            ConvolutionLayer(f32channels, channels, action='leaky'),
            ConvolutionLayer(channels, f32channels, kernel_size=1, padding=0, action='leaky'),
            ConvolutionLayer(f32channels, channels, action='leaky'),
            ConvolutionLayer(channels, f32channels, kernel_size=1, padding=0, action='leaky'),
        )
        self.c3 = nn.Sequential(
            ConvolutionLayer(f32channels, channels, action='leaky'),
            ConvolutionLayer(channels, nc, kernel_size=1, padding=0, action='leaky')
        )

    def forward(self, f64, f32):
        f64 = self.c1(f64)
        y = self.c2(torch.cat([f64, f32], dim=1))
        return y, self.c3(y)


class Predict16(Module):

    def __init__(self, f32_channels, f16_channels, nc):
        super(Predict16, self).__init__()
        self.c1 = ConvolutionLayer(f32_channels, f16_channels, kernel_size=3, stride=2)
        channels = f16_channels + f16_channels
        self.c2 = nn.Sequential(
            ConvolutionLayer(channels, f16_channels, kernel_size=1, padding=0, action='leaky'),
            ConvolutionLayer(f16_channels, channels, action='leaky'),
            ConvolutionLayer(channels, f16_channels, kernel_size=1, padding=0, action='leaky'),
            ConvolutionLayer(f16_channels, channels, action='leaky'),
            ConvolutionLayer(channels, f16_channels, kernel_size=1, padding=0, action='leaky'),
            ConvolutionLayer(f16_channels, channels, action='leaky'),
            ConvolutionLayer(channels, nc, kernel_size=1, padding=0, action='leaky')
        )

    def forward(self, f32, f16):
        f32 = self.c1(f32)
        return self.c2(torch.cat([f32, f16], dim=1))


class MONet(Module):
    """Mask RCNN"""

    def __init__(self):
        super(MONet, self).__init__()
        set = Set()
        depth_multiple = set.depth_multiple
        width_multiple = set.width_multiple
        nc = set.num_category + 8
        self.out_channels = int(3*nc)
        assert self.out_channels%3 ==0
        self.f_size64 = set.image_size//8
        self.f_size32 = set.image_size // 16
        self.f_size16 = set.image_size // 32
        self.block0 = ConvolutionLayer(3, int(32 * width_multiple))
        self.block1 = Block(int(32 * width_multiple), int(64 * width_multiple), max(int(depth_multiple * 1), 1))  # 64
        self.block2 = Block(int(64 * width_multiple), int(128 * width_multiple), max(int(depth_multiple * 2), 1))  # 128
        self.block3 = Block(int(128 * width_multiple), int(256 * width_multiple), max(int(depth_multiple * 8), 1))  # 256
        self.block4 = Block(int(256 * width_multiple), int(512 * width_multiple), max(int(depth_multiple * 8), 1))  # 512
        self.block5 = Block(int(512 * width_multiple), int(1024 * width_multiple), max(int(depth_multiple * 4), 1))  # 1024
        self.spp = nn.Sequential(
            ConvolutionLayer(int(1024 * width_multiple), int(512 * width_multiple), kernel_size=1, padding=0,
                             action='leaky'),
            ConvolutionLayer(int(512 * width_multiple), int(1024 * width_multiple), action='leaky'),
            # ChannelAttention(int(1024 * width_multiple)),
            # SpatialAttention(),
            ConvolutionLayer(int(1024 * width_multiple), int(512 * width_multiple), kernel_size=1, padding=0,
                             action='leaky'),
            Pool(),
            ConvolutionLayer(4 * int(512 * width_multiple), int(512 * width_multiple), kernel_size=1, padding=0,
                             action='leaky'),
            ConvolutionLayer(int(512 * width_multiple), int(1024 * width_multiple), action='leaky'),
            # ChannelAttention(int(1024 * width_multiple)),
            # SpatialAttention(),
            ConvolutionLayer(int(1024 * width_multiple), int(512 * width_multiple), kernel_size=1, padding=0,
                             action='leaky'),
        )
        self.fpn1 = FPN(int(512 * width_multiple), int(256 * width_multiple))
        self.fpn2 = FPN(int(256 * width_multiple), int(128 * width_multiple))
        self.predict64 = nn.Sequential(
            ConvolutionLayer(int(128 * width_multiple), int(256 * width_multiple), action='leaky'),
            ConvolutionLayer(int(256 * width_multiple), int(3*nc), action='leaky')
        )
        self.predict32 = Predict32(int(128 * width_multiple), int(256 * width_multiple), int(3*nc))
        self.predict16 = Predict16(int(256 * width_multiple), int(512 * width_multiple), int(3*nc))

    def forward(self, image):
        f_512 = self.block0(image)
        f_256 = self.block1(f_512)  # out 64 channels
        f_128 = self.block2(f_256)  # out 128 channels
        f_64 = self.block3(f_128)  # out 256 channels
        f_32 = self.block4(f_64)  # out 512 channels
        f_16 = self.block5(f_32)  # out 1024 channels
        f_16 = self.spp(f_16)  # out 512*width_multiple channels
        up_32 = self.fpn1(f_16, f_32)  # out 256 channels  (512*width_multiple)//2
        up_64 = self.fpn2(up_32, f_64)  # out 128 channels (int(512*width_multiple)//2)//2
        p64 = self.predict64(up_64)
        f32, p32 = self.predict32(up_64, up_32)
        p16 = self.predict16(f32, f_16)
        return p64.permute(0, 2, 3, 1).reshape(-1, self.f_size64, self.f_size64, 3, self.out_channels // 3), p32.permute(0, 2, 3, 1).reshape(
            -1, self.f_size32, self.f_size32, 3, self.out_channels // 3), p16.permute(0, 2, 3, 1).reshape(-1, self.f_size16, self.f_size16, 3,
                                                                                    self.out_channels // 3)


if __name__ == '__main__':
    m = MONet()
    import torch
    # torch.save(m.state_dict(),'net.pt')
    # print(m)
    # torch.save(m.st(3, 1, 512, 512)
    # y = m(x)
    # print(y[0].shape, y[1].shape, y[2].shape)
    # exit()ate_dict(), 'mask_rcnn.pt')
    # x = torch.randn
