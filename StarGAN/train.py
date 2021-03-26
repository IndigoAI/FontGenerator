from stargan_dataloader import Dataset
from stargan import Generator, Discriminator
from wandb_config import API_KEY

from torchvision.utils import make_grid
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch.nn as nn
import torch
import wandb
import os

from calculate_fid import calculate_fid
from inception import fid_inception_v3

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def permute_labels(labels, mode='train'):
    if mode == 'train':
        return labels[torch.randperm(len(labels))]
    return labels[:, torch.randperm(labels.shape[1])]


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand((len(real_samples), 1, 1, 1), device=device)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates, _ = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.shape, requires_grad=False, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class StarGANLearner(pl.LightningModule):
    def __init__(self, n_attr, n_unsupervised, gen_params, discr_params, optim_params, lambds, n_critic=5):
        super().__init__()
        self.n_attr = n_attr

        self.G = Generator(**gen_params)
        self.D = Discriminator(**discr_params)
        self.optim_params = optim_params

        # n_unsupervised fonts + 1 dummy id (for supervised)
        self.font_emb = nn.Embedding(n_unsupervised + 1, self.n_attr)  # attribute intensity

        self.lambda_clf = lambds['lambda_clf']
        self.lambda_gp = lambds['lambda_gp']
        self.lambda_rec = lambds['lambda_rec']

        self.clf_loss = nn.BCEWithLogitsLoss()
        self.rec_loss = nn.L1Loss()

        self.steps = 0
        self.n_critic = n_critic
        self.sample_val = None

    def forward(self, img, label):
        return self.G(img, label)

    def training_step(self, batch, batch_idx, optimizer_idx):
        src_image = batch['src_image']
        src_attr = batch['src_attribute']
        src_label = batch['src_label'].unsqueeze(-1)
        src_emb = batch['src_embed']

        # font embeddings bs x 37
        src_emb = self.font_emb(src_emb)
        src_emb = torch.sigmoid(3 * src_emb)

        # if sup - use initial emb, if unsup - use learned embs
        src_emb = src_label * src_attr + (1 - src_label) * src_emb
        src_emb = torch.where(src_emb >= 0.5, torch.tensor(1.).to(self.device), torch.tensor(0.).to(self.device))

        trg_emb = permute_labels(src_emb)

        # forward G
        if optimizer_idx == 0:
            fake_image = self(src_image, trg_emb)
            fake_out_src, fake_out_cls = self.D(fake_image)
            rec_image = self(fake_image, src_emb)

            adv_loss = -fake_out_src.mean()
            clf_loss = self.clf_loss(fake_out_cls, trg_emb)
            rec_loss = self.rec_loss(src_image, rec_image)
            loss_G = adv_loss + self.lambda_clf * clf_loss + self.lambda_rec * rec_loss
            return {'loss': loss_G, 'loss_G': loss_G}

        elif optimizer_idx == 1:
            self.steps += 1
            real_out_src, real_out_cls = self.D(src_image)
            fake_image = self(src_image, trg_emb)
            fake_out_src, fake_out_cls = self.D(fake_image.detach())

            gradient_penalty = compute_gradient_penalty(self.D, src_image.data, fake_image.data, device=self.device)
            adv_loss = real_out_src.mean() - fake_out_src.mean() - self.lambda_gp * gradient_penalty
            clf_loss = self.clf_loss(real_out_cls, src_emb)
            loss_D = -adv_loss + self.lambda_clf * clf_loss
            return {'loss': loss_D, 'loss_D': loss_D}

    def training_epoch_end(self, outputs):
        avg_g_loss = torch.stack([x.get('loss_G') for x in outputs if x.get('loss_G') is not None]).mean()
        avg_d_loss = torch.stack([x.get('loss_D') for x in outputs if x.get('loss_D') is not None]).mean()
        self.logger.log_metrics({'train_g_epoch_loss': avg_g_loss,
                                 'train_d_epoch_loss': avg_d_loss,
                                 'epoch': self.current_epoch})

    def calculate_val_input(self, batch):
        src_image = batch['src_image']
        src_emb = batch['src_embed']

        # source from unsup - use unsup emb
        src_emb = self.font_emb(src_emb)
        src_emb = torch.sigmoid(3 * src_emb)
        src_emb = torch.where(src_emb >= 0.5, torch.tensor(1.).to(self.device), torch.tensor(0.).to(self.device))

        trg_emb = permute_labels(src_emb, mode='val')
        fake_image = self(src_image, trg_emb)
        return fake_image, src_emb

    def validation_step(self, batch, *args):
        src_image = batch['src_image']
        fake_image, src_emb = self.calculate_val_input(batch)

        rec_image = self(fake_image, src_emb)
        loss = self.rec_loss(rec_image, src_image)
    
        if self.sample_val is None:
            self.sample_val = torch.cat((src_image[:10], fake_image[:10]), 0)
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
        step_size = self.optim_params['step_size']
        gamma = self.optim_params['gamma']
        optimizer_G = Adam([
            {'params': self.G.parameters()},
            {'params': self.font_emb.parameters(), 'lr': 1e-3}],
            lr=lr, betas=(beta1, beta2))
        optimizer_D = Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))

        g_scheduler = StepLR(optimizer_G, step_size=step_size, gamma=gamma)
        d_scheduler = StepLR(optimizer_D, step_size=step_size, gamma=gamma)

        return (
            {'optimizer': optimizer_G, 'scheduler': g_scheduler, 'frequency': 1},
            {'optimizer': optimizer_D, 'scheduler': d_scheduler, 'frequency': self.n_critic}
        )


if __name__ == '__main__':
    attribute_path = 'data/attributes.txt'
    image_path = 'data/image/'
    batch_size = 16
    epochs = 500

    n_unsupervised = 968
    n_attr = 37

    gen_params = {
        'c_dim': 37,
        'n_res': 6
    }

    discr_params = {
        'img_size': 64,
        'c_dim': 37,
        'n_hidden': 5
    }

    optim_params = {
        'lr': 1e-4,
        'beta1': 0.5,
        'beta2': 0.99,
        'step_size': 1,
        'gamma': 0.8
    }

    lambds = {
        'lambda_clf': 1,
        'lambda_gp': 10,
        'lambda_rec': 10
    }

    model = StarGANLearner(n_attr, n_unsupervised, gen_params, discr_params, optim_params, lambds)

    train_dataset = Dataset(attribute_path, image_path,  mode='train')
    train_loader = data.DataLoader(dataset=train_dataset,
                                   drop_last=True,
                                   shuffle=True,
                                   batch_size=batch_size)

    val_dataset = Dataset(attribute_path, image_path,  mode='test')
    val_loader = data.DataLoader(dataset=val_dataset,
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

    # Calculate FID
    classifier = fid_inception_v3()

    ckpt_path = 'stargan_epoch=107-val_loss=0.009.ckptstargan_epoch=107-val_loss=0.009.ckpt'
    model = StarGANLearner.load_from_checkpoint(ckpt_path,
                                                n_attr=n_attr,
                                                n_unsupervised=n_unsupervised,
                                                discr_params=discr_params,
                                                optim_params=optim_params,
                                                lambds=lambds)

    fid = calculate_fid(val_loader, model, classifier)
    print(fid)
