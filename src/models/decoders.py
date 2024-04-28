import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import LinearSoftmax


class ProdLDADecoder(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class DVAEDecoder(nn.Module):
    """
    Module that parameterizes the obs likelihood p(x | z)
    """
    def __init__(
        self,
        vocab_size: int,
        num_topics: int, 
        bias_term: bool = True,
        softmax_beta: bool = False,
        beta_init: torch.tensor = None,
    ):
        super().__init__()

        if not softmax_beta:
            self.eta_layer = nn.Linear(num_topics, vocab_size, bias=bias_term)
        else:
            self.eta_layer = LinearSoftmax(num_topics, vocab_size, bias=bias_term)
        
        if beta_init is not None:
            self.eta_layer.weight.data.copy_(beta_init.T)

        # this matches NVDM / TF implementation, which does not use scale
        self.eta_bn_layer = nn.BatchNorm1d(
            vocab_size, eps=0.001, momentum=0.001, affine=True
        )
        self.eta_bn_layer.weight.data.copy_(torch.ones(vocab_size))
        self.eta_bn_layer.weight.requires_grad = False

    def forward(self, z: torch.tensor, bn_annealing_factor: float = 0.0) -> torch.tensor:
        eta = self.eta_layer(z)
        eta_bn = self.eta_bn_layer(eta)

        x_recon = (
            (bn_annealing_factor) * F.softmax(eta, dim=-1)
            + (1 - bn_annealing_factor) * F.softmax(eta_bn, dim=-1)
        )
        return x_recon
    
    @property
    def beta(self) -> np.ndarray:
        return self.eta_layer.weight.T.cpu().detach().numpy()
