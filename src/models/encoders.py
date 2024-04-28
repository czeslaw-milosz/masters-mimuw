import torch
import torch.nn as nn
import torch.nn.functional as F


class ProdLDAEncoder(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale

class ProdLDADirichletEncoder(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout, device):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_topics)
        self.bn = nn.BatchNorm1d(num_topics, affine=False)
        # self.fcmu = nn.Linear(hidden, num_topics)
        # self.fclv = nn.Linear(hidden, num_topics)
        # self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        # self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.hidden = hidden
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.device = device

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        h = self.fc3(h)
        log_concentration = self.bn(h)
        print(log_concentration.shape)
        return log_concentration.exp()


class DVAEEncoder(nn.Module):
    """
    Module that parameterizes the dirichlet distribution q(z|x)
    """
    def __init__(
            self,
            vocab_size: int,
            num_topics: int,
            embeddings_dim: int,
            hidden_dim: int,
            dropout: float,
        ):
        super().__init__()

        # setup linear transformations
        self.embedding_layer = nn.Linear(vocab_size, embeddings_dim)
        self.embed_drop = nn.Dropout(dropout)

        self.second_hidden_layer = hidden_dim > 0
        if self.second_hidden_layer:
            self.fc = nn.Linear(embeddings_dim, hidden_dim)
            self.fc_drop = nn.Dropout(dropout)

        self.alpha_layer = nn.Linear(hidden_dim or embeddings_dim, num_topics)

        # this matches NVDM / TF implementation, which does not use scale
        self.alpha_bn_layer = nn.BatchNorm1d(
            num_topics, eps=0.001, momentum=0.001, affine=True
        )
        self.alpha_bn_layer.weight.data.copy_(torch.ones(num_topics))
        self.alpha_bn_layer.weight.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        embedded = F.relu(self.embedding_layer(x))
        embedded_do = self.embed_drop(embedded)

        hidden_do = embedded_do
        if self.second_hidden_layer:
            hidden = F.relu(self.fc(embedded_do))
            hidden_do = self.fc_drop(hidden)

        alpha = self.alpha_layer(hidden_do)
        alpha_bn = self.alpha_bn_layer(alpha)

        alpha_pos = torch.max(
            F.softplus(alpha_bn),
            torch.tensor(0.00001, device=alpha_bn.device)
        )

        return alpha_pos
