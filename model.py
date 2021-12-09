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
        x = self.embed(x)  # (N, L, d)
        if h is None:
            h = self._init_hidden(x.size()[0], x.device)
        x, h = self.rnn(x, h)  # (N, L, d), (num_layers, N, d)
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
        return loss

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


class AttentionBlock(nn.Module):

    def __init__(self, feat_dim, num_heads, kv_dim, q_dim):
        super(AttentionBlock, self).__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.kv_dim = kv_dim
        self.q_dim = q_dim
        self.linear_q = nn.Linear(feat_dim, q_dim * num_heads, bias=False)
        self.linear_k = nn.Linear(feat_dim, kv_dim * num_heads, bias=False)
        self.linear_v = nn.Linear(feat_dim, kv_dim * num_heads, bias=False)
        self.mlp_out = nn.Sequential(nn.Linear(q_dim * num_heads, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, x, mask):
        """
        Args:
            x: (N, L, feat_d)
            mask: (N, L, L)
        """
        N, L, _ = x.size()
        q = self.linear_q(x).view(N, L, self.num_heads, self.q_dim)
        k = self.linear_k(x).view(N, L, self.num_heads, self.kv_dim)
        v = self.linear_v(x).view(N, L, self.num_heads, self.kv_dim)
        attention_score = (k[:, :, None, :, :] * v[:, None, :, :, :]).sum(-1)  # (N, L, L, n_heads)
        attention_score = torch.where(mask.unsqueeze(-1), attention_score,
                                      torch.full_like(attention_score, float("-inf")))
        attention_score = torch.softmax(attention_score, dim=2)
        x_new = torch.einsum("bijh,bjhk->bihk", attention_score, q).reshape(N, L, -1)
        x = self.layer_norm(x + self.mlp_out(x_new))
        return x


class AttentionModel(nn.Module):

    def __init__(self, config, dict):
        super(AttentionModel, self).__init__()
        self.feat_dim = config.feat_dim
        self.num_layers = config.num_layers
        self.dict = dict
        self.out_dim = len(dict)

        self.embed = nn.Embedding(self.out_dim, self.feat_dim)
        self.blocks = nn.ModuleList([AttentionBlock(config.feat_dim, config.num_heads, config.kv_dim, config.q_dim)
                                     for _ in range(self.num_layers)])
        self.output = nn.Linear(self.feat_dim, self.out_dim)

    @staticmethod
    def _get_pair_mask(N, L):
        index = torch.arange(0, L)
        mask = (index[:, None] >= index[None, :]).view(1, L, L).repeat(N, 1, 1)
        return mask

    def forward(self, x):
        """
        Args:
            x: (N, L)
        """
        N, L = x.size()
        mask = self._get_pair_mask(N, L).to(x.device)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.output(x)

    def get_loss(self, x, mask=None):
        """
        Args:
            x: (N, L)
            mask: None or (N, L)
        """
        ret = self(x)
        logp = F.log_softmax(ret, dim=-1)
        loss = (- logp[:, :-1] * F.one_hot(x[:, 1:], num_classes=self.out_dim)).sum(-1)
        if mask is not None:
            loss = (loss * mask[:, 1:]).sum((-1, -2)) / mask[:, 1:].sum((-1, -2))
        else:
            loss = loss.mean((-1, -2))
        return loss

    def _generate(self):
        x = torch.ones((1, 1), dtype=torch.long) * self.dict.word2idx["<ini>"]
        for _ in range(50):
            prop = self(x)
            x = torch.cat([x, torch.multinomial(prop[:, -1].exp().view(-1), 1).view(1, 1)], dim=-1)
            if x[0, -1].item() == self.dict.word2idx["<end>"]:
                break
        return x

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
