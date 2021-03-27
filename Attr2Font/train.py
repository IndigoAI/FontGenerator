from Attr2Font.dataloader import Dataset
from Attr2Font.attr2font import Generator, Discriminator, CXLoss
from Attr2Font.vgg_cx import VGG19_CX
# from wandb_config import API_KEY

from calculate_fid import calculate_fid
from inception import fid_inception_v3

from torchvision.utils import make_grid
from torch.optim import Adam
from torch.utils import data
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch.nn as nn
import torch
# import wandb
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attr2FontLearner(pl.LightningModule):
    def __init__(self, attr_emb, n_unsupervised, gen_params, discr_params, optim_params, lambds):
        super().__init__()
        self.n_attr = gen_params['n_attr']
        self.return_attr_D = discr_params['return_attr']

        self.G = Generator(**gen_params)
        self.D = Discriminator(**discr_params)
        self.optim_params = optim_params

        # attribute: N x 37 -> N x 37 x 64
        self.attr_emb = nn.Embedding(self.n_attr, attr_emb)
        # n_unsupervised fonts + 1 dummy id (for supervised)
        self.font_emb = nn.Embedding(n_unsupervised + 1, self.n_attr)  # attribute intensity

        self.lambd_adv = lambds['lambd_adv']
        self.lambd_pixel = lambds['lambd_pixel']
        self.lambd_char = lambds['lambd_char']
        self.lambd_cx = lambds['lamdb_cx']
        self.lambd_attr = lambds['lamdb_attr']

        self.gan_loss_fn = nn.MSELoss()
        self.pixel_loss_fn = nn.L1Loss()
        self.char_loss_fn = nn.CrossEntropyLoss()
        self.cx_loss_fn = CXLoss(sigma=0.5)
        self.attr_loss_fn = nn.MSELoss()

        self.vgg19 = VGG19_CX().to(device)
        self.vgg19.load_model('vgg19-dcbb9e9d.pth')
        self.vgg19.eval()
        self.vgg_layers = ['conv3_3', 'conv4_2']

        self.sample_val = None

    def forward(self, src_image, src_style, delta_emb, delta_attr_emb):
        return self.G(src_image, src_style, delta_emb, delta_attr_emb)

    def training_step(self, batch, batch_idx, optimizer_idx):
        src_image = batch['src_image']
        src_char = batch['src_char']
        src_attr = batch['src_attribute']
        src_style = batch['src_style']
        src_label = batch['src_label'].unsqueeze(-1)
        src_emb = batch['src_embed']

        trg_image = batch['trg_image']
        trg_attr = batch['trg_attribute']
        trg_label = batch['trg_label'].unsqueeze(-1)
        trg_emb = batch['trg_embed']

        # numbers from 0 to 36
        attr_ids = torch.tensor([i for i in range(self.n_attr)]).to(device)
        attr_ids = attr_ids.repeat(len(src_image), 1)

        # feature embeddings bs x 37 x emb_size
        src_attr_emb = self.attr_emb(attr_ids)
        trg_attr_emb = self.attr_emb(attr_ids)

        # font embeddings bs x 37
        src_emb = self.font_emb(src_emb)
        src_emb = torch.sigmoid(3 * src_emb)
        trg_emb = self.font_emb(trg_emb)
        trg_emb = torch.sigmoid(3 * trg_emb)

        # if sup - use initial emb, if unsup - use learned embs
        src_unsup_emb = src_label * src_attr + (1 - src_label) * src_emb
        trg_unsup_emb = trg_label * trg_attr + (1 - trg_label) * trg_emb

        # for visual style transformer
        delta_emb = trg_unsup_emb - src_unsup_emb

        # for AAM
        src_unsup_emb = src_unsup_emb.unsqueeze(-1)
        trg_unsup_emb = trg_unsup_emb.unsqueeze(-1)
        src_attr_embd = src_unsup_emb * src_attr_emb
        trg_attr_embd = trg_unsup_emb * trg_attr_emb
        delta_attr_emb = trg_attr_embd - src_attr_embd

        # forward G
        if optimizer_idx == 0:
            trg_fake, src_logits = self(src_image, src_style, delta_emb, delta_attr_emb)
            pred_fake, src_attr_pred, trg_attr_pred = self.D(src_image, trg_fake, trg_unsup_emb)

            adv_loss = self.lambd_adv * self.gan_loss_fn(pred_fake, torch.ones_like(pred_fake))
            pixel_loss = self.lambd_pixel * self.pixel_loss_fn(trg_fake, trg_image)
            char_loss = self.lambd_char * self.char_loss_fn(src_logits, src_char - 10)  # src_char.shape = bs
            attr_loss = self.lambd_attr * (self.attr_loss_fn(src_unsup_emb.squeeze(), src_attr_pred.double()) +
                                           self.attr_loss_fn(trg_unsup_emb.squeeze(), trg_attr_pred.double()))

            cx_loss = torch.zeros(1).to(device)
            if self.lambd_cx > 0:
                vgg_trg_fake = self.vgg19(trg_fake)
                vgg_trg_img = self.vgg19(trg_image)

                for l in self.vgg_layers:
                    cx = self.cx_loss_fn(vgg_trg_img[l], vgg_trg_fake[l])
                    cx_loss += cx * self.lambd_cx

            loss_G = adv_loss + pixel_loss + char_loss + attr_loss + cx_loss
            self.logger.log_metrics({'train_g_step_loss': loss_G.item(),
                                     'train_adv_g_loss': adv_loss.item(),
                                     'train_pixel_loss': pixel_loss.item(),
                                     'train_char_loss': char_loss.item(),
                                     'train_loss_cx': cx_loss.item(),
                                     'train_attr_g_loss': attr_loss.item()})
            return {'loss': loss_G}

        # forward D
        elif optimizer_idx == 1:
            with torch.no_grad():
                trg_fake, _ = self(src_image, src_style, delta_emb, delta_attr_emb)
            pred_real, src_real_attr_pred, trg_real_attr_pred = self.D(src_image, trg_image, trg_unsup_emb.detach())
            pred_fake, src_fake_attr_pred, trg_fake_attr_pred = self.D(src_image, trg_fake, trg_unsup_emb.detach())

            loss_real = self.gan_loss_fn(pred_real, torch.ones_like(pred_real))
            loss_fake = self.gan_loss_fn(pred_fake, torch.zeros_like(pred_real))

            attr_loss = torch.zeros(1).to(device)
            if self.return_attr_D:
                attr_loss = self.lambd_attr * (self.attr_loss_fn(src_unsup_emb.squeeze(), src_real_attr_pred.double()) +
                                               self.attr_loss_fn(trg_unsup_emb.squeeze(), trg_real_attr_pred.double()) +
                                               self.attr_loss_fn(src_unsup_emb.squeeze(), src_fake_attr_pred.double()) +
                                               self.attr_loss_fn(trg_unsup_emb.squeeze(), trg_fake_attr_pred.double()))

            adv_loss = loss_real + loss_fake
            loss_D = adv_loss + attr_loss
            self.logger.log_metrics({'train_d_step_loss': loss_D,
                                     'train_adv_d_loss': adv_loss,
                                     'train_attr_d_loss': attr_loss})
            return {'loss': loss_D}

    def training_epoch_end(self, outputs):
        avg_g_loss = torch.stack([x['loss'] for x in outputs[0]]).mean()
        avg_d_loss = torch.stack([x['loss'] for x in outputs[1]]).mean()

        self.logger.log_metrics({'train_g_epoch_loss': avg_g_loss,
                                 'train_d_epoch_loss': avg_d_loss,
                                 'epoch': self.current_epoch})

    def calculate_val_input(self, batch):
        src_image = batch['src_image']
        src_style = batch['src_style']
        src_emb = batch['src_embed']

        trg_attr = batch['trg_attribute']

        attr_ids = torch.tensor([i for i in range(self.n_attr)]).to(device)
        attr_ids = attr_ids.repeat(len(src_image), 1)

        src_attr_emb = self.attr_emb(attr_ids)
        trg_attr_emb = self.attr_emb(attr_ids)

        # source from unsup - use unsup emb
        src_unsup_emb = self.font_emb(src_emb)
        src_unsup_emb = torch.sigmoid(3 * src_unsup_emb)

        # VST
        delta_emb = trg_attr - src_unsup_emb

        # AAM
        src_attr_embd = src_unsup_emb.unsqueeze(-1) * src_attr_emb
        trg_attr_embd = trg_attr.unsqueeze(-1) * trg_attr_emb
        delta_attr_emb = trg_attr_embd - src_attr_embd
        return src_image, src_style, delta_emb, delta_attr_emb

    def validation_step(self, batch, *args):
        trg_image = batch['trg_image']
        src_image, src_style, delta_emb, delta_attr_emb = self.calculate_val_input(batch)

        trg_fake, _ = self(src_image, src_style, delta_emb, delta_attr_emb)
        loss = self.pixel_loss_fn(trg_fake, trg_image)

        if self.sample_val is None:
            self.sample_val = torch.cat((trg_image[:10], trg_fake[:10]), 0)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        grid_img = make_grid(self.sample_val, nrow=10)
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
            {'params': self.font_emb.parameters(), 'lr': 1e-3}],
            lr=lr, betas=(beta1, beta2))
        optimizer_D = Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))
        return [optimizer_G, optimizer_D], []


if __name__ == '__main__':
    attribute_path = 'data/attributes.txt'
    image_path = 'data/image/'
    batch_size = 16
    epochs = 500

    attr_emb = 64
    n_unsupervised = 968

    gen_params = {
        'in_channels': 3,
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
        'lr': 2e-4,
        'beta1': 0.5,
        'beta2': 0.99
    }

    lambds = {
        'lambd_adv': 5,
        'lambd_pixel': 50,
        'lambd_char': 3,
        'lamdb_cx': 6,
        'lamdb_attr': 20
    }

    model = Attr2FontLearner(attr_emb, n_unsupervised, gen_params, discr_params, optim_params, lambds)

    train_dataset = Dataset(attribute_path, image_path,  mode='train')
    train_loader = data.DataLoader(dataset=train_dataset,
                                   shuffle=True,
                                   drop_last=True,
                                   batch_size=batch_size)

    val_dataset = Dataset(attribute_path, image_path,  mode='test')
    val_loader = data.DataLoader(dataset=val_dataset,
                                 drop_last=True,
                                 batch_size=batch_size)

    # os.environ["WANDB_API_KEY"] = API_KEY
    # os.environ['WANDB_MODE'] = 'dryrun'
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

    # Calculate FID
    classifier = fid_inception_v3()

    ckpt_path = 'epoch=108-val_loss=0.148.ckpt'
    model = Attr2FontLearner.load_from_checkpoint(ckpt_path,
                                                  attr_embd=attr_emb,
                                                  n_unsupervised=n_unsupervised,
                                                  gen_params=gen_params,
                                                  discr_params=discr_params,
                                                  optim_params=optim_params,
                                                  lambds=lambds)

    fid = calculate_fid(val_loader, model.G, classifier)
    print(fid)

