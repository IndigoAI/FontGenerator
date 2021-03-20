from dataloader import Dataset
from model import Generator, Discriminator

from torch.optim import Adam
from torch.utils import data
import pytorch_lightning as pl
import torch.nn as nn


class Attr2FontLearner(pl.LightningModule):
    def __init__(self, attr_emb, n_unsupervised, gen_params, optim_params):
        super().__init__()
        self.G = Generator(**gen_params)
        self.D = Discriminator()
        self.optim_params = optim_params

        # attribute: N x 37 -> N x 37 x 64
        self.attr_emb = nn.Embedding(gen_params['attr_channel'], attr_emb)
        # n_unsupervised fonts + 1 dummy id (for supervised)
        self.font_emb = nn.Embedding(n_unsupervised + 1, gen_params['attr_channel'])  # attribute intensity

        self.pixel_loss_fn = nn.L1Loss()
        self.gan_loss_fn = nn.MSELoss()
        self.char_loss_fn = nn.CrossEntropyLoss()  #
        self.attr_loss_fn = nn.MSELoss()  # why not smooth l1?
        # CX loss

    def training_step(self, batch, *args):
        # font embed
        src_image = batch['src_image']
        src_char = batch['src_char']
        src_attr = batch['src_attribute']
        src_style = batch['src_style']
        src_label = batch['src_label']
        src_embed = batch['src_embed']

        trg_image = batch['trg_image']
        trg_char = batch['trg_char']
        trg_attr = batch['trg_attr']
        trg_label = batch['trg_label']
        trg_embed = batch['trg_embed']

    def configure_optimizers(self):
        lr = self.optim_params['lr']
        beta1 = self.optim_params['beta1']
        beta2 = self.optim_params['beta2']
        optimizer_G = Adam([
            {'params': self.G.parameters()},
            {'params': self.attr_emb.parameters(), 'lr': 1e-3},
            {'params': self.unsup_emb.parameters(), 'lr': 1e-3}],
        lr=lr, betas=(beta1, beta2))
        optimizer_D = Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D


if __name__ == '__main__':
    attribute_path = 'data/attributes.txt'
    image_path = 'data/image/'
    batch_size = 16

    attr_emb = 64
    n_unsupervised = 968

    gen_params = {
        'attr_channel': 37,
        'style_out': 256,
        'out_channels': 3,
        'n_attr': 37,
        'attention': True
    }

    train_dataset = Dataset(attribute_path, image_path,  mode='train')
    train_loader = data.DataLoader(dataset=train_dataset,
                                   drop_last=True,
                                   batch_size=batch_size)

    val_dataset = Dataset(attribute_path, image_path,  mode='test')
    val_loader = data.DataLoader(dataset=train_dataset,
                                 drop_last=True,
                                 batch_size=batch_size)






