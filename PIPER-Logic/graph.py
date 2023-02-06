import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedGCN(nn.Module):
    def __init__(self, hidden_size, activation_ru, activation_x):
        super(GatedGCN, self).__init__()
        if activation_ru == "sigmoid":
            activation_func_ru = nn.Sigmoid()
        elif activation_ru == "tanh":
            activation_func_ru = nn.Tanh()
        elif activation_ru == "ReLU":
            activation_func_ru = nn.ReLU()
        elif activation_ru == "LeakyReLU":
            activation_func_ru = nn.LeakyReLU()
        elif activation_ru == "RReLU":
            activation_func_ru = nn.RReLU()
        elif activation_ru == "softsign":
            activation_func_ru = nn.Softsign()
        elif activation_ru == "softplus":
            activation_func_ru = nn.Softplus()
        elif activation_ru == "GELU":
            activation_func_ru = nn.GELU()
        else:
            raise ("error!")

        if activation_x == "sigmoid":
            activation_func_x = nn.Sigmoid()
        elif activation_x == "tanh":
            activation_func_x = nn.Tanh()
        elif activation_x == "ReLU":
            activation_func_x = nn.ReLU()
        elif activation_x == "LeakyReLU":
            activation_func_x = nn.LeakyReLU()
        elif activation_x == "RReLU":
            activation_func_x = nn.RReLU()
        elif activation_x == "softsign":
            activation_func_x = nn.Softsign()
        elif activation_x == "softplus":
            activation_func_x = nn.Softplus()
        elif activation_x == "GELU":
            activation_func_x = nn.GELU()
        else:
            raise ("error!")

        self.u_linear = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), activation_func_ru)
        self.r_linear = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), activation_func_ru)
        self.x_linear = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), activation_func_x)

    def forward(self, x, adj):
        """
        :param x: (batch_size, seq_len, dim)
        :param adj: (batch_size, seq_len, seq_len)
        :return:
        """
        y = torch.matmul(adj, x)  # (batch_size, seq_len, dim)
        u = self.u_linear(torch.cat([y, x], dim=-1))
        r = self.r_linear(torch.cat([y, x], dim=-1))
        x_bar = self.x_linear(torch.cat([y, r * x], dim=-1))
        x_final = (1 - u) * x + u * x_bar
        return x_final


class DistillModule(nn.Module):
    def __init__(self, hidden_dim, para_attention=True, para_distill=True):
        super(DistillModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.para_attention = para_attention
        self.para_distill = para_distill
        if para_attention:
            self.we = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if para_distill:
            self.w1 = nn.Linear(hidden_dim, hidden_dim)
            self.w2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, mask1=None, mask2=None):
        """
        :param s1: (bsz, len, D)
        :param s2: (bsz, len, D)
        :param mask1: (bsz, len)
        :param mask2: (bsz, len)
        :return:
        """
        batch_size, seq_len, hidden_dim = s1.size()
        if self.para_attention:
            s1 = self.we(s1)
        e = torch.matmul(s1, s2.transpose(-1, -2))

        if mask1 is not None and mask2 is not None:
            mask1 = mask1.unsqueeze(dim=-1).repeat(1, 1, seq_len)
            mask2 = mask2.unsqueeze(dim=-1).repeat(1, 1, seq_len).transpose(1, 2)
            mask = -1e5 * (1 - mask1 * mask2)
            e = e + mask

        et = e.transpose(-1, -2)

        alpha1 = F.softmax(et, dim=-1)
        alpha2 = F.softmax(e, dim=-1)

        x1 = torch.matmul(alpha2, s2)
        x2 = torch.matmul(alpha1, s1)

        d1 = s1 - x1
        d2 = s2 - x2
        if self.para_distill:
            d1 = self.w1(d1)
            d2 = self.w2(d2)
        d1 = torch.sigmoid(d1)
        d2 = torch.sigmoid(d2)

        o1 = s1 * (1. - d1)
        o2 = s2 * (1. - d2)

        n1 = torch.mean(torch.abs(d1))
        n2 = torch.mean(torch.abs(d2))

        return o1, o2, n1, n2