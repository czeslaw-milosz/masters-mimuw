import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import losses
from models.encoders import ProdLDAEncoder, ProdLDADirichletEncoder
from models.decoders import ProdLDADecoder
from config import config

class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout, loss_regularizer=None, reg_lambda=1e+03):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        assert loss_regularizer in (None, "l1", "l2", "cosine")
        self.loss_regularizer = loss_regularizer
        self.reg_lambda = reg_lambda
        self.encoder = ProdLDAEncoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = ProdLDADecoder(vocab_size, num_topics, dropout)

    def model(self, docs):
        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a Softmax-normal distribution
            logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics))
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = F.softmax(logtheta, -1)

            # conditional distribution of 𝑤𝑛 is defined as 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            count_param = self.decoder(theta)
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                'obs',
                dist.Multinomial(total_count, count_param),
                obs=docs
            )

    def guide(self, docs):
        pyro.module("encoder", self.encoder)

        if self.loss_regularizer == "l2":
            pyro.factor("beta_penalty", self.reg_lambda * losses.l2_penalty(self.decoder.beta.weight), has_rsample=True)
        elif self.loss_regularizer == "l1":
            pyro.factor("beta_penalty", self.reg_lambda * losses.l1_penalty(self.decoder.beta.weight), has_rsample=True)
        elif self.loss_regularizer == "cosine":
            pyro.factor("beta_penalty", self.reg_lambda * losses.cosine_penalty(self.decoder.beta.weight), has_rsample=True)

        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution, where μ and Σ are the encoder outputs
            logtheta_loc, logtheta_scale = self.encoder(docs)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T
    

class ProdLDADirichlet(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = ProdLDADirichletEncoder(vocab_size, num_topics, hidden, dropout, device)
        self.decoder = ProdLDADecoder(vocab_size, num_topics, dropout)

    def model(self, docs):
        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0], subsample_size=config.BATCH_SIZE):
            batch_docs = pyro.subsample(docs, event_dim=1)
            theta = pyro.sample(
                "theta", dist.Dirichlet(torch.ones(self.num_topics)).to_event(1)
            )
            logits = self.decoder(theta)
            # count_param = self.decoder(theta)
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                'obs', dist.Multinomial(total_count, logits=logits), obs=batch_docs
            )

    def guide(self, docs):
        pyro.module("encoder", self.encoder)
        with pyro.plate("documents", docs.shape[0], subsample_size=config.BATCH_SIZE):
            batch_docs = pyro.subsample(docs, event_dim=1)
            concentration = self.encoder(batch_docs)
            theta = pyro.sample(
                "theta", dist.Dirichlet(concentration).to_event(1)
            )

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T
