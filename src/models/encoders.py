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

        # h = F.softplus(self.fc1(inputs))
        # h = F.softplus(self.fc2(h))
        # h = self.drop(h)
        # # μ and Σ are the outputs
        # logtheta_loc = self.bnmu(self.fcmu(h))
        # logtheta_logvar = self.bnlv(self.fclv(h))
        # logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        # return logtheta_loc, logtheta_scale
