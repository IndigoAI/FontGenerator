from Attr2Font.train import Attr2FontLearner
from StarGAN.train import StarGANLearner
import numpy as np
import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import ipywidgets as widgets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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



def load_model(weights, model='Attr2Font'):
    if model == 'Attr2Font':
        model = Attr2FontLearner.load_from_checkpoint(weights,
                                                      attr_emb=64,
                                                      n_unsupervised=968, gen_params=gen_params,
                                                      discr_params=discr_params, optim_params=optim_params,
                                                      lambds=lambds).to(device)
    return model


def generate(target, model, dataloader, idx=-1):
    """
    idx - index of batch in dataloader, if idx = -1 - random
    """
    target = torch.tensor(target).to(device) / 100

    with torch.no_grad():
        if idx == -1:
            idx = torch.randint(len(dataloader), (1, 1))
        for i, batch in enumerate(dataloader):
            if i == idx:
                source = batch
                break

        attr_ids = torch.tensor([i for i in range(37)]).to(device)
        attr_ids = attr_ids.repeat(52, 1)
        embedding = model.attr_emb(attr_ids)

        src_attr_embd = source['src_attribute'].unsqueeze(-1).to(device) * embedding
        trg_attr_embd = target.unsqueeze(-1) * embedding

        delta_attr_emb = trg_attr_embd - src_attr_embd
        delta_emb = target - source['src_attribute'].to(device)
        gen, _ = model(source['src_image'].to(device), source['src_style'].to(device), delta_emb, delta_attr_emb)
        gen = make_grid(gen).permute(1, 2, 0)

        return make_grid(source['src_image']).permute(1, 2, 0), gen.cpu()



def get_widgets(attribute_path='data/attributes.txt', values=0):
    """
    values - if values = 0 all values in widget 0, else - random
    """
    with open(attribute_path, 'r') as file:
        names = file.readline().split()[1:]

    widget = []
    if values:
        values = torch.randint(100, (1, len(names))).squeeze()
    else:
        values = torch.zeros(len(names))

    for name, value in zip(names, values):
        w = widgets.FloatSlider(
            value=value,
            min=0,
            max=100,
            step=0.1,
            description=name,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        widget.append(w)

    return widget, widgets.VBox(widget, layout=widgets.Layout(flex_flow='row wrap'))

def show(image):
    if len(image) == 1:
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.show()
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 10))
        axes[0].imshow(image[0])
        axes[0].set_title('Real', fontsize=30)
        axes[1].imshow(image[1])
        axes[1].set_title('Fake', fontsize=30)
        plt.show()

