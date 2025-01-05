import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math

from models.backbone import build_backbone
from models.transformer import build_transformer
from utils.misc import clean_state_dict
import torch.nn.functional as F

CLASS_15 = {
    0: [24, 27],
    1: [5],
    2: [13],
    3: [11],
    4: [3, 7, 8, 9, 19],
    5: [4, 6, 20, 21],
    6: [14],
    7: [15, 16],
    8: [2, 22, 23],
    9: [0],
    10: [17, 25],
    11: [10],
    12: [1],
    13: [12],
    14: [18, 26],
}


CLASS_9 = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6],
    4: [7, 8, 9],
    5: [10],
    6: [11],
    7: [12],
    8: [13, 14]
}


def one_hot_embedding(labels, num_classes=10, device=None):
    # Convert to One Hot Encoding
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()

    # Convert to One Hot Encoding
    labels = labels.long()  # Ensure labels are of type 'long' (integer)
    y = torch.eye(num_classes).to(device)
    return y[labels]

# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    device = torch.device('cuda:0')
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = p
    label, = map(torch.tensor, (label,))
    label = label.to(device)
    alpha = alpha.to(device)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (torch.mean(A + B))

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class UCFN(nn.Module):
    def __init__(self, backbone, transfomer, num_class, dataname):
        """[summary]
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (28 for Intentonomy).
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class
        self.dataname = dataname
        self.device = torch.device('cuda:0')
        hidden_dim = transfomer.d_model
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc_num_class = GroupWiseLinear(num_class, hidden_dim, bias=True)
        self.fc_middle = GroupWiseLinear(15, hidden_dim, bias=True)
        self.fc_coarse = GroupWiseLinear(9, hidden_dim, bias=True)

    def DS_Combin_two(self, alpha1, alpha2, classes):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        device = self.device
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        alpha[0] = alpha[0].to(device)
        alpha[1] = alpha[1].to(device)
        b, S, E, u = dict(), dict(), dict(), dict()

        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = classes/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def forward(self, input, y, global_step):
        device = self.device
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]
        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, pos)[0] # B,K,d
        lambda_epochs = 50
        y_hat = []
        
        y_hat.append(hs[0])
        y_hat.append(hs[1])
        y_hat.append(hs[2])
        total_loss = torch.zeros(3)
        total_loss = total_loss.to(self.device)
        total_prob = []
        total_uncertainty = []
        total_alpha_a = []
        total_t_y_hat_a = []
        alpha_a = torch.zeros([15, len(y_hat[0]), 2])
        alpha_a[:,:,0] = 9999999
        alpha_a = alpha_a.to(device)
        alpha_b = torch.zeros([9, len(y_hat[0]), 2])
        alpha_b[:,:,0] = 9999999
        alpha_b = alpha_b.to(device)
        for i in range(3):           
            tuple_y_hat = torch.split(y_hat[i], 2, dim=1)
            pred_prob = []
            t_y_hat_a = []
            uncertainty_list = []
            for idx, t_y_hat in enumerate(tuple_y_hat):
                t_y = y[i][:, idx]
                t_y_hot = one_hot_embedding(t_y, 2, self.device)
                evidence = torch.relu(t_y_hat)
                alpha = evidence + 1
                if i == 2:
                    alpha = self.DS_Combin_two(alpha, alpha_b[idx], 2)
                if i == 1:
                    for q in range(len(CLASS_9)):
                        if idx in CLASS_9[q]:
                            for b in range(len(alpha)):
                                alpha_b[q][b][0] = torch.min(alpha_b[q][b][0].clone(), alpha[b][0]) 
                                alpha_b[q][b][1] = torch.max(alpha_b[q][b][1].clone(), alpha[b][1])
                    alpha = self.DS_Combin_two(alpha, alpha_a[idx], 2)
                if i == 0:
                    for q in range(len(CLASS_15)):
                        if idx in CLASS_15[q]:
                            for b in range(len(alpha)):
                                alpha_a[q][b][0] = torch.min(alpha_a[q][b][0].clone(), alpha[b][0])
                                alpha_a[q][b][1] = torch.max(alpha_a[q][b][1].clone(), alpha[b][1])
                total_loss[i] += ce_loss(t_y_hot.float(), alpha, 2, global_step, lambda_epochs)
                prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
                uncertainty = 2 / torch.sum(alpha, dim=1, keepdim=True)
                sig_prob = prob[:,1].unsqueeze(-1)
                pred_prob.append(sig_prob)
                sig_t_y_hat = t_y_hat[:,1].unsqueeze(-1)
                t_y_hat_a.append(sig_t_y_hat)
                uncertainty_list.append(uncertainty)
            total_prob.append(torch.cat(pred_prob, dim=1))
            total_t_y_hat_a.append(torch.cat(t_y_hat_a, dim=1))
            total_uncertainty.append(torch.cat(uncertainty_list, dim=1))
        return total_t_y_hat_a, total_loss[0], (total_loss[1]/15)*28, (total_loss[2]/9)*28


def build_UCFN(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = UCFN(
        backbone = backbone,
        transfomer = transformer,
        num_class = 56,
        dataname = args.dataname
    )
    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    return model
        
        