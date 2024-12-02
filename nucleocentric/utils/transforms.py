import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import default_collate
import numbers


def max_proj(img, axis=0, keepdims=False):
    img = img.max(axis=axis, keepdims=keepdims)
    if isinstance(img, torch.Tensor):
        img = img.values
    return img


def unsqueeze_to_ndim(img, n_dim):
    if len(img.shape) < n_dim:
        img = torch.unsqueeze(img,0) if isinstance(img, torch.Tensor) else np.expand_dims(img,0)
        unsqueeze_to_ndim(img, n_dim)
    return img


def squeeze_to_ndim(img, n_dim):
    if len(img.shape) > n_dim:
        img = torch.squeeze(img) if isinstance(img, torch.Tensor) else np.squeeze(img)
        unsqueeze_to_ndim(img, n_dim)
    return img


def get_padding(image,shape=None):    
    _, h, w = image.shape
    if shape is None:
        shape = (np.max([h, w]),np.max([h, w]))
    h_padding = (shape[0] - h) / 2
    v_padding = (shape[1] - w) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))
    return padding


def my_collate(batch):
    batch = list(filter(lambda x : not x[0].isnan().any(), batch))
    return default_collate(batch)


class ShapePad(object):
    def __init__(self, shape=(128,128), fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.shape = shape
        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return torch.nn.functional.pad(img, get_padding(img,self.shape), mode=self.padding_mode,value=self.fill)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class SquarePad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return torch.nn.functional.pad(img, get_padding(img), mode=self.padding_mode,value=self.fill)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class AugmentContrast(object):
    def __init__(self, contrast_range, channel_dim=0, preserve_range=True, per_channel=True, p_per_channel=0.5):
        self.contrast_range = contrast_range
        self.channel_dim = channel_dim
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
    
    def __call__(self, img):
        n_channels = img.shape[self.channel_dim]
        r1, r2 = self.contrast_range
        shape = torch.ones(img.ndim, dtype=int).tolist()
        if self.per_channel:
            shape[self.channel_dim] = n_channels
            factor = (r1 - r2) * torch.rand(shape) + r2
        else:
            factor = (r1 - r2) * torch.rand(shape) + r2
            shape[self.channel_dim] = n_channels
            factor = factor.repeat(shape)
        
        m = img.min()
        M = img.max()
        axis = list(range(img.ndim))
        axis.remove(self.channel_dim)
        augment_channel = torch.rand(shape) <= self.p_per_channel
        factor = torch.where(augment_channel, factor, torch.ones(shape))
        img = (img - img.mean(dim=axis, keepdim=True))*factor + img.mean(dim=axis, keepdim=True)
        if not self.preserve_range:
            m = img.min()
            M = img.max()
        img = img.clip(min=m, max=M)
        return img

class AugmentBrightness(object):
    def __init__(self, mu, sigma, channel_dim=0, preserve_range=True, per_channel=True, p_per_channel=0.5):
        self.mu = mu
        self.sigma = sigma
        self.channel_dim = channel_dim
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
    
    def __call__(self, img):
        n_channels = img.shape[self.channel_dim]
        shape = torch.ones(img.ndim, dtype=int).tolist()
        if self.per_channel:
            shape[self.channel_dim] = n_channels
            rnd_nb = torch.randn(shape)*self.sigma + self.mu
        else:
            rnd_nb = torch.randn(shape)*self.sigma + self.mu
            shape[self.channel_dim] = n_channels
            rnd_nb = rnd_nb.repeat(shape)
        augment_channel = torch.rand(shape) <= self.p_per_channel
        m = img.min()
        M = img.max()
        img = img + augment_channel*rnd_nb
        if not self.preserve_range:
            m = img.min()
            M = img.max()
        img = img.clip(min=m, max=M)
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class ToTensorPerChannel(object):
    def __init__(self):
        pass

    def __call__(self, img):
        n_channels = img.shape[-1]
        out = torch.zeros(np.moveaxis(img, -1, 0).shape)
        for c in range(n_channels):
            out[c,...] = F.to_tensor(img[...,c]).div(img[...,c].max())
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class NormalizeTensorPerChannel(object):
    def __init__(self, pmin, pmax, channel_dim=0, clip=True):
        self.pmin = pmin
        self.pmax = pmax
        self.channel_dim=channel_dim
        self.clip = clip
    
    def __call__(self, img):
        axis = list(range(img.ndim))
        axis.remove(self.channel_dim)
        pmin_values = torch.Tensor(np.percentile(img, self.pmin, axis=axis, keepdims=True).astype(np.float32))
        pmax_values = torch.Tensor(np.percentile(img, self.pmax, axis=axis, keepdims=True).astype(np.float32))
        img = (img - pmin_values)/(pmax_values-pmin_values)
        if self.clip:
            img = img.clip(min=0, max=1)
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class SelectChannels(object):
    def __init__(self, channels2use) -> None:
         self.channels2use = channels2use

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): image from which channels should be selected.
        Returns:
            img (torch.Tensors): image with only the requested channels.
        """
        n_channels = img.shape[0]
        if len(self.channels2use) > n_channels:
            raise ValueError(f'The number of requested channels (channels2use = {self.channels2use}) ' \
                f'exceeds the number of channels present in the image ({n_channels}).')
        return img[np.r_[self.channels2use],:,:]
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"