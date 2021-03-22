from dataloader import Dataset
from model import Generator, Discriminator
from wandb_config import API_KEY

from torchvision.utils import make_grid
from torch.optim import Adam
from torch.utils import data
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch.nn as nn
import torch
import wandb
import os


class Attr2FontLearner(pl.LightningModule):
    def __init__(self, attr_emb, n_unsupervised, gen_params, discr_params, optim_params, lambds):
        super().__init__()
        self.n_attr = gen_params['attr_channel']
        self.return_attr_D = discr_params['return_attr']

        self.G = Generator(**gen_params)
        self.D = Discriminator(**discr_params)
        self.optim_params = optim_params

        # attribute: N x 37 -> N x 37 x 64
        self.attr_emb = nn.Embedding(gen_params['attr_channel'], attr_emb)
        # n_unsupervised fonts + 1 dummy id (for supervised)
        self.font_emb = nn.Embedding(n_unsupervised + 1, gen_params['attr_channel'])  # attribute intensity

        self.lambd_adv = lambds['lambd_avd']
        self.lambd_pixel = lambds['lambd_pixel']
        self.lambd_char = lambds['lambd_char']
        self.lambd_cx = lambds['lamdb_cx']
        self.lambd_attr = lambds['lamdb_attr']

        self.gan_loss_fn = nn.MSELoss()
        self.pixel_loss_fn = nn.L1Loss()
        self.char_loss_fn = nn.CrossEntropyLoss()
        self.cx_loss_fn = None
        self.attr_loss_fn = nn.MSELoss()

        self.sample_val = None

    def training_step(self, batch, batch_idx, optimizer_idx):
        src_image = batch['src_image']
        src_char = batch['src_char']
        src_attr = batch['src_attribute']
        src_style = batch['src_style']
        src_label = batch['src_label']
        src_emb = batch['src_embed']

        trg_image = batch['trg_image']
        trg_attr = batch['trg_attr']
        trg_label = batch['trg_label']
        trg_emb = batch['trg_embed']

        # numbers from 0 to 36
        attr_ids = torch.tensor([i for i in range(self.n_attr)])
        attr_ids = attr_ids.repeat(len(batch), 1)

        # feature embeddings bs x 37 x emb_size
        src_attr_emb = self.attr_emb(attr_ids)
        trg_attr_emb = self.attr_emb(attr_ids)

        # font embeddings bs x 37
        src_emb = self.font_emb(src_emb)
        # src_emb = src_emb.view(src_emb.size(0), src_emb.size(2))  why 3 dims?????
        src_emb = torch.sigmoid(3 * src_emb)      # why 3 ????
        trg_emb = self.font_emb(trg_emb)
        # trg_emb = trg_emb.view(trg_emb.size(0), trg_emb.size(2))
        trg_emb = torch.sigmoid(3 * trg_emb)      # why 3 ????

        # if sup - use initial emb, if unsup - use learned embs
        src_unsup_emb = src_label * src_attr + (1 - src_label) * src_emb
        trg_unsup_emb = trg_label * trg_attr + (1 - trg_label) * trg_emb

        # for visual style transformer
        delta_emb = trg_unsup_emb - src_unsup_emb

        # for AAM
        # src_emb = src_emb.unsqueeze(-1)
        # trg_emb = trg_emb.unsqueeze(-1)
        src_attr_embd = src_unsup_emb * src_attr_emb
        trg_attr_embd = trg_unsup_emb * trg_attr_emb
        delta_attr_emb = trg_attr_embd - src_attr_embd

        # forward G
        if optimizer_idx == 0:
            trg_fake, src_logits = self.G(src_image, src_style, delta_emb, delta_attr_emb)
            pred_fake, src_attr_pred, trg_attr_pred = self.D(src_image, trg_fake, trg_unsup_emb)
            src_vgg = None
            trg_vgg = None

            adv_loss = self.lambd_gan * self.gan_loss_fn(pred_fake, torch.ones_like(pred_fake))
            pixel_loss = self.lambd_pixel * self.pixel_loss_fn(trg_fake, trg_image)
            char_loss = self.lambd_char * self.char_loss_fn(src_logits, src_char)  # src_char.shape = bs
            attr_loss = self.lambd_attr * (self.attr_loss_fn(src_unsup_emb, src_attr_pred) +
                                           self.attr_loss_fn(trg_unsup_emb, trg_attr_pred))

            # cx_loss = self.lambd_cx * self.cx_loss_fn()
            # CX loss
            # cx_loss = torch.zeros(1).to(device)
            # if opts.lambda_cx > 0:
            #     for l in vgg_layers:
            #         cx = cx_loss_fn(vgg_img_B[l], vgg_fake_B[l])
            #         cx_loss += cx * self.lambd_cx
            loss_G = adv_loss + pixel_loss + char_loss + attr_loss  # + cx_loss
            self.logger.log_metrics({'train_g_step_loss': loss_G,
                                     'train_adv_g_loss': adv_loss,
                                     'train_pixel_loss': pixel_loss,
                                     'train_char_loss': char_loss,
                                     'train_loss_cx': cx_loss,
                                     'train_attr_g_loss': attr_loss})
            return {'g_loss': loss_G}

        # forward D
        elif optimizer_idx == 1:
            with torch.no_grad():
                trg_fake, _ = self.G(src_image, src_style, delta_emb, delta_attr_emb)
            pred_real, src_real_attr_pred, trg_real_attr_pred = self.D(src_image, trg_image, trg_unsup_emb.detach())
            pred_fake, src_fake_attr_pred, trg_fake_attr_pred = self.D(src_image, trg_fake, trg_unsup_emb.detach())

            loss_real = self.gan_loss_fn(pred_real, torch.ones_like(pred_real))
            loss_fake = self.gan_loss_fn(pred_fake, torch.zeros_like(pred_real))

            attr_loss = torch.zeros(1)
            if self.return_attr_D:
                attr_loss = self.lambd_attr * (self.attr_loss_fn(src_unsup_emb, src_real_attr_pred) +
                                               self.attr_loss_fn(trg_unsup_emb, trg_real_attr_pred) +
                                               self.attr_loss_fn(src_unsup_emb, src_fake_attr_pred) +
                                               self.attr_loss_fn(trg_unsup_emb, trg_fake_attr_pred))

            adv_loss = loss_real + loss_fake
            loss_D = adv_loss + attr_loss
            self.logger.log_metrics({'train_d_step_loss': loss_D,
                                     'train_adv_d_loss': adv_loss,
                                     'train_attr_d_loss': attr_loss})
            return {'d_loss': loss_D}

    def training_epoch_end(self, outputs):
        avg_g_loss = torch.stack([x['g_loss'] for x in outputs]).mean()
        avg_d_loss = torch.stack([x['d_loss'] for x in outputs]).mean()

        self.logger.log_metrics({'train_g_epoch_loss': avg_g_loss,
                                 'train_d_epoch_loss': avg_d_loss,
                                 'epoch': self.current_epoch})

    def validation_step(self, batch, *args):
        src_image = batch['src_image']
        src_style = batch['src_style']
        src_emb = batch['src_embed']

        trg_image = batch['trg_image']
        trg_sup_emb = batch['trg_embed']

        attr_ids = torch.tensor([i for i in range(self.n_attr)])
        attr_ids = attr_ids.repeat(len(batch), 1)

        src_attr_emb = self.attr_emb(attr_ids)
        trg_attr_emb = self.attr_emb(attr_ids)

        # source from unsup - use unsup emb
        src_unsup_emb = self.font_emb(src_emb)
        # src_emb = src_emb.view(src_emb.size(0), src_emb.size(2))  why 3 dims?????
        src_unsup_emb = torch.sigmoid(3 * src_unsup_emb)  # why 3 ????

        # VST
        delta_emb = trg_sup_emb - src_unsup_emb

        # AAM
        src_attr_embd = src_unsup_emb * src_attr_emb
        trg_attr_embd = trg_sup_emb * trg_attr_emb
        delta_attr_emb = trg_attr_embd - src_attr_embd

        trg_fake, _ = self.G(src_image, src_style, delta_emb, delta_attr_emb)
        loss = self.pixel_loss_fn(trg_fake, trg_image)

        if self.sample_val is None:
            self.sample_val = torch.cat((trg_image[:10], trg_fake[:10]), 1)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        grid_img = make_grid(self.sample_val, nrow=2)
        self.logger.log_metrics({'val_loss': avg_loss,
                                 'epoch': self.current_epoch,
                                 'val imgs': [wandb.Image(grid_img)]})
        self.sample_val = None
        return {'val_loss': avg_loss}

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
        return [optimizer_G, optimizer_D], []


if __name__ == '__main__':
    attribute_path = 'data/attributes.txt'
    image_path = 'data/image/'
    batch_size = 16
    epochs = 100

    attr_emb = 64
    n_unsupervised = 968

    gen_params = {
        'attr_channel': 37,
        'style_out': 256,
        'out_channels': 3,
        'n_attr': 37,
        'attention': True
    }

    discr_params = {
        'in_channels': 3,
        'attr_channels': 37,
        'return_attr': True
    }

    optim_params = {
        'lr': None,
        'beta1': None,
        'beta2': None
    }

    lambds = {
        'lambd_gan': None,
        'lambd_pixel': None,
        'lambd_char': None,
        'lamdb_cx': None,
        'lamdb_attr': None
    }

    model = Attr2FontLearner(attr_emb, n_unsupervised, gen_params, discr_params, optim_params, lambds)

    train_dataset = Dataset(attribute_path, image_path,  mode='train')
    train_loader = data.DataLoader(dataset=train_dataset,
                                   drop_last=True,
                                   batch_size=batch_size)

    val_dataset = Dataset(attribute_path, image_path,  mode='test')
    val_loader = data.DataLoader(dataset=train_dataset,
                                 drop_last=True,
                                 batch_size=batch_size)

    os.environ["WANDB_API_KEY"] = API_KEY
    os.environ['WANDB_MODE'] = 'dryrun'
    wandb_logger = WandbLogger(project='Attr2Font')

    gpus = torch.cuda.device_count()
    accelerator = 'ddp' if gpus == 2 else None
    saving_ckpt = ModelCheckpoint(dirpath='checkpoints',
                                  filename='{epoch}-{val_loss:.3f}',
                                  save_top_k=3,
                                  monitor='val_loss',
                                  verbose=True)

    trainer = pl.Trainer(max_epochs=epochs,
                         gpus=gpus,
                         accelerator=accelerator,
                         logger=wandb_logger,
                         checkpoint_callback=saving_ckpt)

    trainer.fit(model, train_loader, val_loader)
