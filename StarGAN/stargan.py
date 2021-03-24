import torch
from torch import nn
from utils import compute_gradient_penalty
from torch.optim.lr_scheduler import StepLR



class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.res = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
                         nn.InstanceNorm2d(in_channels),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
                         nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.res(x)


class Generator(nn.Module):
    def __init__(self, c_dim=5, n_res=6):
        super().__init__() # c_dim = n_dim
        modules = [
            # Initial convolution block
            nn.Conv2d(3 + c_dim, 64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsampling
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        
        # Residual blocks
        for _ in range(n_res):
            modules.append(ResidualBlock(256))
            
        modules.extend([
            # Upsampling
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.Conv2d(64, 3, 7, stride=1, padding=3, bias=False),
            nn.Tanh()
        ])

        self.layers = nn.Sequential(*modules)
        
    def forward(self, x, labels):
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        labels = labels.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, labels], dim=1)
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, img_size, c_dim, n_hidden=5):
        super().__init__()

        modules = []
        in_d = 3
        out_d = 64
        for _ in range(n_hidden + 1):
            modules.append(nn.Conv2d(in_d, out_d, 4, stride=2, padding=1))
            modules.append(nn.LeakyReLU(0.01, inplace=True))
            in_d = out_d
            out_d = out_d * 2

        self.layers = nn.Sequential(*modules)
        self.out1 = nn.Conv2d(in_d, 1, 3, stride=1, padding=1, bias=False)
        kernel = int(img_size / (2 ** (n_hidden + 1)))
        self.out2 = nn.Conv2d(in_d, c_dim, kernel, bias=False)


    def forward(self, x):
        h = self.layers(x)
        out_src = self.out1(h)
        out_cls = self.out2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

    
class StarGAN:
    def __init__(self, params):
        self.p = params
        self.G = Generator(self.p['n_dim'], self.p['n_res'])
        self.D = Discriminator(self.p['img_size'], self.p['n_dim'], self.p['n_hidden'])
        self.g_optim = torch.optim.Adam(self.G.parameters(), self.p['g_lr'], [self.p['b1'], self.p['b2']])
        self.d_optim = torch.optim.Adam(self.D.parameters(), self.p['d_lr'], [self.p['b1'], self.p['b2']])
        self.g_scheduler = StepLR(self.g_optim, step_size=1, gamma=0.8)
        self.d_scheduler = StepLR(self.d_optim, step_size=1, gamma=0.8)
        self.clf_loss = nn.BCEWithLogitsLoss()
        self.rec_loss = nn.L1Loss()
        self.device = None


    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def to(self, device):
        self.D.to(device)
        self.G.to(device)
        self.device = device

    def save(self, file):
        torch.save({
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'g_optim': self.g_optim.state_dict(),
            'd_optim': self.d_optim.state_dict()
        }, file)

    def load(self, file):
        checkpoint = torch.load(file)
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.g_optim.load_state_dict(checkpoint['g_optim'])
        self.d_optim.load_state_dict(checkpoint['d_optim'])

    def trainD(self, real_image, src_label, trg_label):
        self.d_optim.zero_grad()

        real_out_src, real_out_cls = self.D(real_image)
        fake_image = self.G(real_image, trg_label)
        fake_out_src, fake_out_cls = self.D(fake_image.detach())

        gradient_penalty = compute_gradient_penalty(self.D, real_image.data, fake_image.data)
        adv_loss = real_out_src.mean() - fake_out_src.mean() - self.p['lambda_gp'] * gradient_penalty
        clf_loss = self.clf_loss(real_out_cls, src_label)
        loss =  - adv_loss + self.p['lambda_clf'] * clf_loss

        loss.backward()
        self.d_optim.step()

        return adv_loss, clf_loss, loss

    def trainG(self, real_image, src_label, trg_label):
        self.g_optim.zero_grad()

        fake_image = self.G(real_image, trg_label)
        fake_out_src, fake_out_cls = self.D(fake_image)
        rec_image = self.G(fake_image, src_label)

        adv_loss = - fake_out_src.mean()
        clf_loss = self.clf_loss(fake_out_cls, trg_label)
        rec_loss = self.rec_loss(real_image, rec_image)
        loss = adv_loss + self.p['lambda_clf'] * clf_loss + self.p['lambda_rec'] * rec_loss

        loss.backward()
        self.g_optim.step()

        return adv_loss, clf_loss, rec_loss, loss

    def generate(self, image, label):
        return self.G(image, label)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optim.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optim.param_groups:
            param_group['lr'] = d_lr

