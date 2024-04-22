import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

from models import losses
from models.encoders import ProdLDAEncoder, ProdLDADirichletEncoder
from models.decoders import ProdLDADecoder
from config import config


class ClassicLDA:
    def __init__(self, max_iter=100, num_topics=10, evaluate_every=10, random_seed=2137) -> None:
        self.max_iter = max_iter
        self.num_topics = num_topics
        self.evaluate_every = evaluate_every
        self.random_seed = random_seed
        self.model = LatentDirichletAllocation(
            n_components=self.num_topics,
            max_iter=self.max_iter,
            learning_method="batch",
            learning_offset=10.0,
            perp_tol=1e-3,
            random_state=self.random_seed,
            n_jobs=-1,
            evaluate_every=self.evaluate_every
        )

    def beta(self, normalized=False) -> np.ndarray:
        return self.model.components_ if not normalized else self.model.components_ / self.model.components_.sum(axis=1, keepdims=True)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout, loss_regularizer=None, reg_lambda=1e+03) -> None:
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
    def __init__(self, vocab_size, num_topics, hidden, dropout, device) -> None:
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


class RRTVAE(nn.Module):

    def __init__(self, vocab_size, num_topics, hidden_size=500, 
                 lambda_=0.01, delta=1e10, prior_alpha=1.0, device="cuda:0") -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.hidden_size = hidden_size
        self.lambda_ = lambda_
        self.delta = delta
        self.prior_alpha = torch.Tensor(1, num_topics).fill_(prior_alpha)
        self.device = device
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.vocab_size, self.hidden_size),   
            nn.ReLU(True),
            
            nn.Linear(self.hidden_size, self.hidden_size),   
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(True),
            
            nn.Linear(self.hidden_size, self.num_topics),
            nn.BatchNorm1d(self.num_topics),
        )
        
        # Decoder
        self.decoder = nn.Linear(self.num_topics, self.vocab_size)             
        self.decoder_bn = nn.BatchNorm1d(self.vocab_size)
        self.decoder.weight.data.uniform_(0, 1)

    def beta(self):
        return self.decoder.weight.cpu().detach().T 

    def RealSampler(self, parameter, multi=False):
        m = torch.distributions.dirichlet.Dirichlet(parameter)
        data = m.sample((2000,)) if multi else m.sample()
        return data.to(self.device)
            
    # Sampling from Dirichlet distributions using RRT
    def RRT(self, parameter):
        # Round the Dirichlet parameter to its delta decimal place
        param_round = torch.floor(self.delta * parameter) / self.delta
        # Sampling from a "Rounded" Dirichlet distribution
        sample = self.RealSampler(param_round, multi=False)
        # Construct the target sample
        sample = sample + (parameter - param_round) * self.lambda_
        sample = sample / torch.sum(sample, dim=1, keepdim=True)
        return sample
        
    
    def forward(self, inputs, avg_loss=True):
        # Encoder
        alpha = self.encoder(inputs)
        alpha = torch.exp(alpha/4)
        alpha = F.hardtanh(alpha, min_val=0., max_val=30)
        # Sampling using RRT
        p = self.RRT(alpha)
        # Decoder
        recon = F.softmax(self.decoder_bn(self.decoder(p)), dim=1)  # Reconstruct a distribution over vocabularies
        return recon, self.loss(inputs=inputs, recon=recon, alpha=alpha, avg=avg_loss)
      

    def loss(self, inputs, recon, alpha, avg=True):
        # Negative log-likelihood
        NLL  = -(inputs * (recon + 1e-10).log()).sum(1)
        # Dirichlet prior
        prior_alpha = self.prior_alpha.expand_as(alpha).to("cuda:0")
        # KL divergence between two Dirichlet distributions
        KL = torch.mvlgamma(alpha.sum(1), p=1) - torch.mvlgamma(alpha, p=1).sum(1) - torch.mvlgamma(prior_alpha.sum(1), p=1) + torch.mvlgamma(prior_alpha, p=1).sum(1) + ((alpha - prior_alpha) * (torch.digamma(alpha) - torch.digamma(alpha.sum(dim=1, keepdim=True).expand_as(alpha)))).sum(1) 
        # loss
        loss = (NLL + KL)
        # In the training mode, return averaged loss. In the testing mode, return individual loss
        return (loss.mean(), KL.mean()) if avg else (loss, KL)
