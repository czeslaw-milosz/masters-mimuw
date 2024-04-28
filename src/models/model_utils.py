import numpy as np
import pyro
import pyro.distributions as dist
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO, JitTraceMeanField_ELBO
from pyro.optim import Adam

from typing import Dict, List, Union


class L1RegularizedTraceMeanField_ELBO(pyro.infer.TraceMeanField_ELBO):
    def __init__(self, *args, l1_params=None, l1_weight=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.l1_params = l1_params
        self.l1_weight = l1_weight

    @staticmethod
    def l1_regularize(param_names, weight):
        params = torch.cat([pyro.param(p).view(-1) for p in param_names])
        return weight * torch.norm(params, 1)

    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss_standard = self.differentiable_loss(model, guide, *args, **kwargs)
        loss = loss_standard + self.l1_regularize(self.l1_params, self.l1_weight)

        loss.backward()
        loss = loss.item()

        pyro.util.warn_if_nan(loss, "loss")
        return loss


class CollapsedMultinomial(dist.Multinomial):
    """
    Equivalent to n separate `MultinomialProbs(probs, 1)`, where `self.log_prob` treats each
    element of `value` as an independent one-hot draw (instead of `MultinomialProbs(probs, n)`)
    """
    def log_prob(self, value: torch.tensor) -> torch.tensor:
        return ((self.probs + 1e-10).log() * value).sum(-1)


class LinearSoftmax(nn.Linear):
    """
    Linear layer where the weights are first put through a softmax
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, F.softmax(self.weight, dim=0), self.bias)


class NPMI:
    def __init__(
        self,
        bin_ref_counts: Union[np.ndarray, scipy.sparse.spmatrix],
        vocab: Dict[str, int] = None,
    ):
        assert bin_ref_counts.max() == 1
        self.bin_ref_counts = bin_ref_counts
        if scipy.sparse.issparse(self.bin_ref_counts):
            self.bin_ref_counts = self.bin_ref_counts.tocsc()
        self.npmi_cache = {} # calculating NPMI is somewhat expensive, so we cache results
        self.vocab = vocab

    def compute_npmi(
        self,
        beta: np.ndarray = None,
        topics: Union[np.ndarray, List] = None,
        vocab: Dict[str, int] = None,
        n: int = 10
    ) -> np.ndarray:
        """
        Compute NPMI for an estimated beta (topic-word distribution) parameter using
        binary co-occurence counts from a reference corpus

        Supply `vocab` if the topics contain terms that first need to be mapped to indices
        """
        if beta is not None and topics is not None:
            raise ValueError(
                "Supply one of either `beta` (topic-word distribution array) "
                "or `topics`, a list of index or word lists"
        )
        if vocab is None and any([isinstance(idx, str) for idx in topics[0][:n]]):
            raise ValueError(
                "If `topics` contains terms, not indices, you must supply a `vocab`"
            )
    
        if beta is not None:
            topics = np.flip(beta.argsort(-1), -1)[:, :n]
        if topics is not None:
            topics = [topic[:n] for topic in topics]
        if vocab is not None:
            assert(len(vocab) == self.bin_ref_counts.shape[1])
            topics = [[vocab[w] for w in topic[:n]] for topic in topics]

        num_docs = self.bin_ref_counts.shape[0]
        npmi_means = []
        for indices in topics:
            npmi_vals = []
            for i, idx_i in enumerate(indices):
                for idx_j in indices[i+1:]:
                    ij = frozenset([idx_i, idx_j])
                    try:
                        npmi = self.npmi_cache[ij]
                    except KeyError:
                        col_i = self.bin_ref_counts[:, idx_i]
                        col_j = self.bin_ref_counts[:, idx_j]
                        c_i = col_i.sum()
                        c_j = col_j.sum()
                        if scipy.sparse.issparse(self.bin_ref_counts):
                            c_ij = col_i.multiply(col_j).sum()
                        else:
                            c_ij = (col_i * col_j).sum()
                        if c_ij == 0:
                            npmi = 0.0
                        else:
                            npmi = (
                                (np.log(num_docs) + np.log(c_ij) - np.log(c_i) - np.log(c_j)) 
                                / (np.log(num_docs) - np.log(c_ij))
                            )
                        self.npmi_cache[ij] = npmi
                    npmi_vals.append(npmi)
            npmi_means.append(np.mean(npmi_vals))

        return np.array(npmi_means)
