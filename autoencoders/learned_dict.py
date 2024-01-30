from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn
from torchtyping import TensorType

from autoencoders.ensemble import DictSignature

_n_dict_components, _activation_size, _batch_size = None, None, None


class LearnedDict(ABC):
    n_feats: int
    activation_size: int

    @abstractmethod
    def get_learned_dict(self) -> TensorType["_n_dict_components", "_activation_size"]:
        pass

    @abstractmethod
    def encode(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_n_dict_components"]:
        pass

    @abstractmethod
    def to_device(self, device):
        pass

    def decode(self, code: TensorType["_batch_size", "_n_dict_components"]) -> TensorType["_batch_size", "_activation_size"]:
        learned_dict = self.get_learned_dict()
        x_hat = torch.einsum("nd,bn->bd", learned_dict, code)
        return x_hat

    def predict(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_activation_size"]:
        c = self.encode(batch)
        x_hat = self.decode(c)
        return x_hat

    def n_dict_components(self):
        return self.get_learned_dict().shape[0]


class Identity(LearnedDict):
    def __init__(self, activation_size):
        self.n_feats = activation_size
        self.activation_size = activation_size

    def get_learned_dict(self):
        return torch.eye(self.n_feats)

    def encode(self, batch):
        return batch

    def to_device(self, device):
        pass


class IdentityReLU(LearnedDict):
    def __init__(self, activation_size, bias: Optional[torch.Tensor] = None):
        self.n_feats = activation_size
        self.activation_size = activation_size
        if bias:
            self.bias = bias
        else:
            self.bias = torch.zeros(activation_size)
        assert self.bias.shape == (activation_size,)

    def get_learned_dict(self):
        return torch.eye(self.n_feats)

    def encode(self, batch):
        return torch.clamp(batch + self.bias, min=0.0)

    def to_device(self, device):
        self.bias = self.bias.to(device)


class RandomDict(LearnedDict):
    def __init__(self, activation_size, n_feats=None):
        if not n_feats:
            n_feats = activation_size
        self.n_feats = n_feats
        self.activation_size = activation_size
        self.encoder = torch.randn(n_feats, activation_size)
        self.encoder_bias = torch.zeros(n_feats)

    def get_learned_dict(self):
        return self.encoder

    def encode(self, batch):
        c = torch.einsum("nd,bd->bn", self.encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)


class UntiedSAE(LearnedDict):
    def __init__(self, encoder, decoder, encoder_bias):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_bias = encoder_bias
        self.n_feats, self.activation_size = self.encoder.shape

    def get_learned_dict(self):
        norms = torch.norm(self.decoder, 2, dim=-1)
        return self.decoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

    def set_grad(self):
        self.encoder.requires_grad = True
        self.encoder_bias.requires_grad = True
        self.decoder.requires_grad = True

    def encode(self, batch):
        c = torch.einsum("nd,bd->bn", self.encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c
    
class AnthropicSAE(LearnedDict):
    def __init__(self, encoder, decoder, encoder_bias, shift_bias):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_bias = encoder_bias
        self.shift_bias = shift_bias
        self.n_feats, self.activation_size = self.encoder.shape

    def get_learned_dict(self):
        norms = torch.norm(self.decoder, 2, dim=-1)
        return self.decoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)
        self.shift_bias = self.shift_bias.to(device)
        
    def set_grad(self):
        self.encoder.requires_grad = True
        self.encoder_bias.requires_grad = True
        self.shift_bias.requires_grad = True
        
        self.decoder.requires_grad = True

    def encode(self, batch):
        batch = batch - self.shift_bias
        c = torch.einsum("nd,bd->bn", self.encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c
    
    def decode(self, code):
        learned_dict = self.get_learned_dict()
        x_hat = torch.einsum("nd,bn->bd", learned_dict, code)
        return x_hat + self.shift_bias

class TransferSAE(LearnedDict):
    def __init__(self, autoencoder, decoder, decoder_bias=None, scale=None, mode="free"):
        """
        mode: "scale" (restrict scaling factor), 
        "rotation" (restrict rotations of dictionary), 
        "bias" (restrict bias shift),
        "free" (train everything), 
        """
        assert mode in ["noscale", "norotation", "nobias", "free", "scale", "enorotation", "efree"], "mode not of right type"  # represents what gets frozen
        self.mode = mode
        self.encoder = autoencoder.encoder
        self.encoder_bias = autoencoder.encoder_bias
        self.shift_bias = autoencoder.shift_bias
        self.n_feats, self.activation_size = self.encoder.shape
        
        self.decoder = decoder
        
        if decoder_bias is None:
            self.decoder_bias = autoencoder.shift_bias
        else:
            self.decoder_bias = decoder_bias
        
        if scale is None:
            self.scale = torch.ones_like(self.encoder_bias)
        else:
            self.scale = scale

    def get_learned_dict(self):
        norms = torch.norm(self.decoder, 2, dim=-1)
        return self.decoder / torch.clamp(norms, 1e-8)[:, None]
    
    def get_feature_scales(self):
        return self.scale

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)
        self.shift_bias = self.shift_bias.to(device)
        self.decoder_bias = self.decoder_bias.to(device)
        self.scale.requires_grad = self.scale.to(device)
    
    def set_grad(self):
        self.encoder.requires_grad = False
        self.encoder_bias.requires_grad = False
        self.shift_bias.requires_grad = False
        
        self.decoder.requires_grad = True
        self.decoder_bias.requires_grad = True
        self.scale.requires_grad = True

        if self.mode=="noscale":
            self.scale.requires_grad = False

        if self.mode=="norotation":
            self.decoder.requires_grad=False

        if self.mode=="nobias":
            self.decoder_bias.requires_grad=False

        if self.mode=="free": # we don't freeze anything if "free"
            pass
        
        if self.mode=="scale":            
            self.decoder.requires_grad = False
            self.decoder_bias.requires_grad = False
            self.scale.requires_grad = True
        
        if self.mode=="enorotation":
            self.encoder_bias.requires_grad = True
            
            self.decoder.requires_grad = False
            self.decoder_bias.requires_grad = False
            self.scale.requires_grad = True
        
        if self.mode=="efree":
            self.encoder_bias.requires_grad = True
            self.encoder.requires_grad = True
            
            self.decoder.requires_grad = False
            self.decoder_bias.requires_grad = False
            self.scale.requires_grad = True
    
    def parameters(self):
        params = []
        for param in [self.encoder, self.encoder_bias, self.shift_bias, self.decoder,self.decoder_bias, self.scale]:
            if param.requires_grad:
                params.append(param)
        return params

    def encode(self, batch):
        batch = batch - self.shift_bias
        c = torch.einsum("nd,bd->bn", self.encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c
        
    def decode(self, code):
        learned_dict = self.get_learned_dict()
        scaled_features = code * self.scale # element-wise scale
        x_hat = torch.einsum("nd,bn->bd", learned_dict, scaled_features)
        return x_hat + self.decoder_bias
    
    def apply_scale(self, code):
        return code * self.scale

    def decode_post_scale(self, scaled_features):
        learned_dict = self.get_learned_dict()
        x_hat = torch.einsum("nd,bn->bd", learned_dict, scaled_features)
        return x_hat + self.decoder_bias

class TiedSAE(LearnedDict):
    def __init__(self, encoder, encoder_bias, norm_encoder=False):
        self.encoder = encoder
        self.encoder_bias = encoder_bias
        self.norm_encoder = norm_encoder
        self.n_feats, self.activation_size = self.encoder.shape

    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

    def set_grad(self):
        self.encoder.requires_grad = True
        self.encoder_bias.requires_grad = True

    def encode(self, batch):
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        c = torch.einsum("nd,bd->bn", encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c


class ReverseSAE(LearnedDict):
    """This is the same as a tied SAE, but we reverse the bias if the feature activation is non-zero before the decoder matrix"""

    def __init__(self, encoder, encoder_bias, norm_encoder=False):
        self.encoder = encoder
        self.encoder_bias = encoder_bias
        self.norm_encoder = norm_encoder
        self.n_feats, self.activation_size = self.encoder.shape

    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

    def encode(self, batch):
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        c = torch.einsum("nd,bd->bn", encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c

    def decode(self, c):
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        feat_is_on = c > 0.0
        c[feat_is_on] = c[feat_is_on] - self.encoder_bias.repeat(c.shape[0], 1)[feat_is_on]
        x_hat = torch.einsum("dn,bn->bd", encoder, c)
        return x_hat


class AddedNoise(LearnedDict):
    def __init__(self, noise_mag, activation_size, device=None):
        self.noise_mag = noise_mag
        self.activation_size = activation_size
        self.device = "cpu" if device is None else device

    def get_learned_dict(self):
        return torch.eye(self.activation_size, device=self.device)

    def to_device(self, device):
        self.device = device

    def encode(self, batch):
        noise = torch.randn(batch.shape[0], self.activation_size, device=batch.device) * self.noise_mag
        return batch + noise


class Rotation(LearnedDict):
    def __init__(self, matrix, device=None):
        self.matrix = matrix
        self.activation_size = matrix.shape[0]
        self.device = "cpu" if device is None else device

        self.matrix = self.matrix.to(self.device)

    def get_learned_dict(self):
        return self.matrix

    def to_device(self, device):
        self.matrix = self.matrix.to(device)
        self.device = device

    def encode(self, batch):
        return torch.einsum("nd,bd->bn", self.matrix, batch)
