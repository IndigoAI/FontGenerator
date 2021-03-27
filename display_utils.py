from Attr2Font.train import Attr2FontLearner
from StarGAN.train import StarGANLearner
from config import PARAMS
import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import ipywidgets as widgets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights, model='Attr2Font'):
    assert model in ['Attr2Font', 'StarGAN'], 'Unknown model'
    if model == 'Attr2Font':
        params = PARAMS[model]
        model = Attr2FontLearner.load_from_checkpoint(weights,
                                                      attr_emb=64,
                                                      n_unsupervised=968,
                                                      **params).to(device)
    else:
        params = PARAMS[model]
        model = StarGANLearner.load_from_checkpoint(weights,
                                                    n_attr=37,
                                                    n_unsupervised=968,
                                                    **params)
    return model


def generate(target, model, dataloader, idx=-1, model_name='Attr2Font'):
    """
    idx - index of batch in dataloader, if idx = -1 - random
    """
    assert model_name in ['Attr2Font', 'StarGAN'], 'Unknown model'

    target = torch.tensor(target).to(device) / 100


    with torch.no_grad():
        if idx == -1:
            idx = torch.randint(len(dataloader), (1, 1))
        for i, batch in enumerate(dataloader):
            if i == idx:
                source = batch
                break

        if model_name == 'Attr2Font':
            attr_ids = torch.tensor([i for i in range(37)]).to(device)
            attr_ids = attr_ids.repeat(52, 1)
            embedding = model.attr_emb(attr_ids)

            src_attr_embd = source['src_attribute'].unsqueeze(-1).to(device) * embedding
            trg_attr_embd = target.unsqueeze(-1) * embedding

            delta_attr_emb = trg_attr_embd - src_attr_embd
            delta_emb = target - source['src_attribute'].to(device)
            gen, _ = model(source['src_image'].to(device), source['src_style'].to(device), delta_emb, delta_attr_emb)
            gen = make_grid(gen).permute(1, 2, 0)
        else:
            target = target.repeat(52, 1)
            gen = model(source['src_image'].to(device), target)
            gen = make_grid(gen).permute(1, 2, 0)

        return make_grid(source['src_image']).permute(1, 2, 0), gen.cpu()


def get_widgets(values=0):
    """
    values - if values = 0 all values in widget 0, else - random
    """
    attribute_path = 'data/attributes.txt'
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


def get_attribute(model, dataloader, idx=-1):
    attribute_path = 'data/attributes.txt'
    with open(attribute_path, 'r') as file:
        names = file.readline().split()[1:]

    with torch.no_grad():
        if idx == -1:
            idx = torch.randint(len(dataloader), (1, 1))
        for i, batch in enumerate(dataloader):
            if i == idx:
                source = batch
                break


    embed_id = source['src_embed'][0]
    attribute = torch.sigmoid(3 * model.font_emb(torch.tensor(embed_id).to(device))) * 100
    result = {name: attr for name, attr in zip(names, attribute)}
    return source, result

