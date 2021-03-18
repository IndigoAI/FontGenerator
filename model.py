import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_channel),
        )

    def forward(self, x):
        return x + self.block(x)


class Down(nn.Module):
    def __init__(self, in_channel, out_channel, normalize=True, attention=False,
                 lrelu=False, dropout=0.0, bias=False, kernel_size=4, stride=2, padding=1):
        super(Down, self).__init__()
        layers = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=bias)]
        # if attention:
        #     layers.append(SelfAttention(out_channel))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channel))
        if lrelu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class StyleEncoder(nn.Module):
    def __init__(self, in_channel=3, n_style=4, out_channel=256, n_attr=37,
                 res_blocks=8, attention=True):
        super(StyleEncoder, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel * n_style, 64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 512, dropout=0.5),
            Down(512, 512, dropout=0.5),
            Down(512, out_channel, normalize=False, dropout=0.5)
        )

        res_block = []
        for _ in range(res_blocks):
            res_block += [ResidualBlock(out_channel + n_attr)]
        self.res_block = nn.Sequential(*res_block)

    def forward(self, x, attr_difference):
        x = self.downsample(x)
        x = x.squeeze(2)
        attr_difference = attr_difference.unsqueeze(2)
        s = torch.cat([x, attr_difference], 1)
        s = s.unsqueeze(2)
        s_hat = self.res_block(s)
        return s_hat


class Generator(nn.Module):
    def __init__(self, in_channel, style_out=256, n_attr=37):
        super(Generator, self).__init__()
        self.style_encoder = StyleEncoder()

        self.conv = nn.Sequential(nn.Conv2d(in_channel, 64, 7, stride=1, padding=3, bias=False),
                                  nn.InstanceNorm2d(64),
                                  nn.ReLU(inplace=True))

        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512, dropout=0.5)
        self.down5 = Down(512, 512, dropout=0.5)
        self.down6 = Down(512, 512, normalize=False, dropout=0.5)
        self.fc = Down(512, 52, normalize=False, lrelu=False, kernel_size=1, stride=1, padding=0)

        style_channel = style_out + n_attr
        self.skip1 = nn.Sequential(nn.Conv2d(512 + style_channel, 512, 3, stride=1, padding=1, bias=False),
                                        nn.InstanceNorm2d(512),
                                        nn.ReLU(inplace=True))
        self.skip2 = nn.Sequential(nn.Conv2d(512 + style_channel, 512, 3, stride=1, padding=1, bias=False),
                                        nn.InstanceNorm2d(512),
                                        nn.ReLU(inplace=True))
        self.skip3 = nn.Sequential(nn.Conv2d(256 + style_channel, 256, 3, stride=1, padding=1, bias=False),
                                        nn.InstanceNorm2d(256),
                                        nn.ReLU(inplace=True))
        self.skip4 = nn.Sequential(nn.Conv2d(128 + style_channel, 128, 3, stride=1, padding=1, bias=False),
                                        nn.InstanceNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.skip5 = nn.Sequential(nn.Conv2d(64 + style_channel, 64, 3, stride=1, padding=1, bias=False),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))

    def forward(self, src_image, src_style, attr_difference, attr_feature):
        s_hat = self.style_encoder(src_style, attr_difference)
        conv = self.conv(src_image)
        c1 = self.down1(conv)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        c4 = self.down4(c3)
        c5 = self.down5(c4)
        c6 = self.down6(c5)
        logits = self.fc(c6).squeeze(2).squeeze(2)

        h = torch.cat([c6, s_hat], 1)

        s_tile_1 = s_hat.repeat(1, 1, c5.size(2), c5.size(3))
        h1 = torch.cat([c5, s_tile_1], 1)
        h1 = self.skip1(h1)
        # g1 =
        print(s_hat.shape, c5.shape)


if __name__ == '__main__':
    image = torch.ones((16, 3, 64, 64))
    style = torch.ones((16, 12, 64, 64))
    attr = torch.ones((16, 37))
    # model = StyleEncoder()
    generator = Generator(3)
    generator(image, style, attr, None)
    # print(model(x, y).shape)