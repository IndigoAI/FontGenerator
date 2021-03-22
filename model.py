import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

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
        super().__init__()
        layers = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=bias)]
        if attention:
            layers.append(SelfAttention(out_channel))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channel))
        if lrelu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ChannelAttnLayer(nn.Module):
    def __init__(self, channels, channels_ratio=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(channels, channels // channels_ratio, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channels // channels_ratio, channels, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        w = self.conv(self.pool(x))
        return x * w


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, attr_channels, attr_down_ratio, attention=False, dropout=0):
        super().__init__()
        # upsample fmaps from prev step
        img_layers = []
        img_layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False))
        if attention:
            img_layers.append(SelfAttention(out_channels))
        img_layers.append(nn.InstanceNorm2d(out_channels))
        img_layers.append(nn.ReLU(inplace=True))
        if dropout:
            img_layers.append(nn.Dropout(dropout))
        self.img_block = nn.Sequential(*img_layers)

        # conv with channel attention for attrs (F_ca to get alpha star)
        attr_layers = []
        for _ in range(attr_down_ratio):
            attr_layers += [nn.Conv2d(attr_channels, attr_channels, 4, stride=2, padding=1, bias=True),
                            ChannelAttnLayer(attr_channels)]
            if attention:
                attr_layers += [SelfAttention(attr_channels)]
            attr_layers += [nn.ReLU(inplace=True)]
        self.attr_block = nn.Sequential(*attr_layers)

        # conv with with channel attention for img+attr
        img_attr_layers = []
        img_attr_layers += [nn.Conv2d(attr_channels + out_channels, out_channels, 3, padding=1, bias=False),
                            nn.InstanceNorm2d(out_channels),
                            nn.ReLU(inplace=True)]
        img_attr_layers += [ChannelAttnLayer(out_channels),
                            ChannelAttnLayer(out_channels),  # ???
                            # SelfAttention(out_channels),
                            nn.ReLU(inplace=True)]
        self.img_attr_block = nn.Sequential(*img_attr_layers)

    def forward(self, x, attr_vec, h):
        x = self.img_block(x)

        attr_vec = attr_vec.unsqueeze(-1)   # beta
        attr_vec_T = torch.transpose(attr_vec, 2, 3)
        attr_fmap = torch.matmul(attr_vec, attr_vec_T)  # gamma
        attr = self.attr_block(attr_fmap.float())

        img_attr = torch.cat([x, attr], 1)
        img_attr = self.img_attr_block(img_attr)

        out = torch.cat([img_attr, h], 1)
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation=None):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C X (W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out


class StyleEncoder(nn.Module):
    def __init__(self, in_channel=3, n_style=4, out_channel=256, n_attr=37,
                 res_blocks=8, attention=False):
        super().__init__()

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
        attr_difference = attr_difference.unsqueeze(2).float()
        s = torch.cat([x, attr_difference], 1)
        s = s.unsqueeze(2)
        s_hat = self.res_block(s)
        return s_hat


class Generator(nn.Module):
    def __init__(self, in_channels, style_out=256, out_channels=3, n_attr=37, attention=True):
        super().__init__()
        self.style_encoder = StyleEncoder()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, 64, 7, stride=1, padding=3, bias=False),
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

        self.up1 = Up(512 + style_channel, 512, n_attr, 5, dropout=0.5)
        self.up2 = Up(1024, 512, n_attr, 4, dropout=0.5, attention=attention)
        self.up3 = Up(1024, 256, n_attr, 3, attention=attention)
        self.up4 = Up(512, 128, n_attr, 2, attention=attention)
        self.up5 = Up(256, 64, n_attr, 1)

        self.out_conv = nn.Sequential(nn.Upsample(scale_factor=2),
                                      nn.ZeroPad2d((1, 0, 1, 0)),
                                      nn.Conv2d(128, out_channels, 4, padding=1),
                                      nn.Tanh())

    def forward(self, src_image, src_style, attr_difference, attr_feature):
        s_hat = self.style_encoder(src_style, attr_difference)
        conv = self.conv(src_image)
        c1 = self.down1(conv)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        c4 = self.down4(c3)
        c5 = self.down5(c4)
        c6 = self.down6(c5)
        logits = self.fc(c6).squeeze()

        h = torch.cat([c6, s_hat], 1)

        s_tile_1 = s_hat.repeat(1, 1, c5.size(2), c5.size(3))
        h1 = self.skip1(torch.cat([c5, s_tile_1], 1))
        g1 = self.up1(h, attr_feature, h1)

        s_tile_2 = s_hat.repeat(1, 1, c4.size(2), c4.size(3))
        h2 = self.skip2(torch.cat([c4, s_tile_2], 1))
        g2 = self.up2(g1, attr_feature, h2)

        s_tile_3 = s_hat.repeat(1, 1, c3.size(2), c3.size(3))
        h3 = self.skip3(torch.cat([c3, s_tile_3], 1))
        g3 = self.up3(g2, attr_feature, h3)

        s_tile_4 = s_hat.repeat(1, 1, c2.size(2), c2.size(3))
        h4 = self.skip4(torch.cat([c2, s_tile_4], 1))
        g4 = self.up4(g3, attr_feature, h4)

        s_tile_5 = s_hat.repeat(1, 1, c1.size(2), c1.size(3))
        h5 = self.skip5(torch.cat([c1, s_tile_5], 1))
        g5 = self.up5(g4, attr_feature, h5)

        return self.out_conv(g5), logits


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.01))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttrClassifier(nn.Module):
    def __init__(self, in_channels, attr_channels):
        super().__init__()

        self.net = nn.Sequential(DiscriminatorBlock(in_channels, 64, normalize=False),
                                 DiscriminatorBlock(64, 128),
                                 DiscriminatorBlock(128, 256),
                                 DiscriminatorBlock(256, 256),
                                 nn.ZeroPad2d((1, 0, 1, 0)))

        self.out0 = nn.Sequential(nn.Conv2d(256, 256, 4, padding=1, bias=True),
                                    nn.InstanceNorm2d(256),
                                    nn.LeakyReLU(0.01))

        self.out1 = nn.Linear(256 * 4 * 4, attr_channels, True)

    def forward(self, img):
        out = self.net(img)
        out = self.out0(out)
        out = out.view(out.size(0), out.size(1) * out.size(2) * out.size(3))
        out = self.out1(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, attr_channels=37, return_attr=True):
        super().__init__()
        self.return_attr = return_attr

        self.net = nn.Sequential(DiscriminatorBlock(in_channels * 2, 64, normalize=False),
                                 DiscriminatorBlock(64, 128),
                                 DiscriminatorBlock(128, 256),
                                 DiscriminatorBlock(256, 256),
                                 nn.ZeroPad2d((1, 0, 1, 0)))

        self.out_prob = nn.Conv2d(256, 1, 4, padding=1, bias=False)

        if return_attr:
            self.out_cls = AttrClassifier(in_channels, attr_channels)

    def forward(self, src_image, trg_image, trg_emb):
        if self.return_attr:
            input = torch.cat([src_image, trg_image], dim=1)
            out = self.net(input)
            out_real = self.out_prob(out)
            src_attr = self.out_cls(src_image)
            trg_attr = self.out_cls(trg_image)
            return out_real, src_attr, trg_attr
        else:
            trg_emb = trg_emb.repeat(1, 1, src_image.size(2), src_image.size(3))
            input = torch.cat([trg_image, trg_emb], dim=1)
            return self.model(input), None, None


class CXLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0, similarity="consine"):
        super().__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCXHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        # [0] means get the value, torch min will return the index as well
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''

        # print("featureT target size:", featureT.shape)
        # print("featureI inference size:", featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX



if __name__ == '__main__':
    image = torch.ones((16, 3, 64, 64))
    style = torch.ones((16, 12, 64, 64))
    attr = torch.ones((16, 37))
    attr_features = torch.ones((16, 37, 64))

    # generator = Generator(3)
    # generator(image, style, attr, attr_features)

    discr = Discriminator()
    discr(image, image, None)

