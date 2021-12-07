import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BasicModel(nn.Module):

    def __init__(self, config, dict):
        super(BasicModel, self).__init__()

        self.type = config.type
        self.feat_dim = config.feat_dim
        self.num_layers = config.num_layers
        self.dict = dict
        self.out_dim = len(dict)

        self.dropout = nn.Dropout(config.dropout)
        self.embed = nn.Embedding(self.out_dim, self.feat_dim)
        self.rnn = getattr(nn, config.type)(self.feat_dim, self.feat_dim, config.num_layers, dropout=config.dropout,
                                            batch_first=True)
        self.output = nn.Linear(self.feat_dim, self.out_dim)

    def _init_hidden(self, N, device='cpu'):
        if self.type == 'LSTM':
            return (torch.zeros((self.num_layers, N, self.feat_dim)).to(device),
                    torch.zeros((self.num_layers, N, self.feat_dim)).to(device))
        else:
            return torch.zeros((self.num_layers, N, self.feat_dim)).to(device)

    def forward(self, x, h=None):
        """
        Args:
            x: (N, L)
            h: None or (num_layers, N, d)
        """
        x = self.embed(x) # (N, L, d)
        if h is None:
            h = self._init_hidden(x.size()[0], x.device)
        x, h = self.rnn(x, h) # (N, L, d), (num_layers, N, d)
        x = self.output(x)
        return x, h

    def get_loss(self, x, h=None, mask=None):
        """
        Args:
            x: (N, L)
            h: None or (num_layers, N, d)
            mask: None or (N, L)
        """
        ret, h_new = self(x, h)
        logp = F.log_softmax(ret, dim=-1)
        loss = (- logp[:, :-1] * F.one_hot(x[:, 1:], num_classes=self.out_dim)).sum(-1)
        if mask is not None:
            loss = (loss * mask[:, 1:]).sum((-1, -2)) / mask[:, 1:].sum((-1, -2))
        else:
            loss = loss.mean((-1, -2))
        return loss, ret, h_new

    def _generate(self):
        h = self._init_hidden(1)
        x = torch.ones((1, 1), dtype=torch.long) * self.dict.word2idx["<ini>"]
        ret = torch.ones((1, 1), dtype=torch.long) * self.dict.word2idx["<ini>"]
        for _ in range(50):
            prop, h = self(x, h)
            x = torch.multinomial(prop.exp().view(-1), 1).view(1, 1)
            ret = torch.cat([ret, x], dim=-1)
            if x.item() == self.dict.word2idx["<end>"]:
                break
        return ret

    def generate(self, N):
        """
        Args:
            N: Int
        """
        ret = ""
        for _ in range(N):
            seq = self.dict.translate(self._generate().view(-1))
            ret = ret + "\n" + seq
        return ret