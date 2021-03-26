import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm


def resize(image, size=299):
    return nn.functional.interpolate(image, size=size)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(dataloader, model, classifier, out_dim):
    classifier.eval()
    model.eval()
    device = next(model.parameters()).device
    classifier.to(device)

    i = 0
    act_gen = np.zeros((len(dataloader.dataset), out_dim))
    act_real = np.zeros((len(dataloader.dataset), out_dim))
    
    for batch in tqdm(dataloader, desc='activation stats loop', leave=True):
        img = img.to(device)
        label = label.to(device)
        bs = img.shape[0]

        with torch.no_grad():
            gen, _ = model(*batch)

        out_gen = classifier(resize(gen)).cpu().numpy()
        out_real = classifier(resize(img)).cpu().numpy()

        act_gen[i: i + bs] = out_gen
        act_real[i: i + bs] = out_real
        i += bs

    mu1 = np.mean(act_gen, 0)
    sigma1 = np.cov(act_gen, rowvar=False)
    mu2 = np.mean(act_real, 0)
    sigma2 = np.cov(act_real, rowvar=False)
    return mu1, mu2, sigma1, sigma2

@torch.no_grad()
def calculate_fid(dataloader, model, classifier, out_dim=1008):
    
    m1, m2, s1, s2 = calculate_activation_statistics(dataloader, model, classifier, out_dim)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()
