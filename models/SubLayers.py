import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np


class STAttnGraphConv(nn.Module):
    def __init__(self, n_route, n_his, d_attribute, d_out, dis_mat, n_head=4, d_q=32, d_k=64, d_c=10, kt=2, normal=False) -> None:
        super(STAttnGraphConv, self).__init__()
        self.K_S = nn.Parameter(torch.empty(n_head, d_k))
        nn.init.xavier_uniform_(self.K_S.data)
        self.V_S = nn.Parameter(torch.empty(n_head, n_route, d_c))
        nn.init.xavier_uniform_(self.V_S.data)
        self.K_T = nn.Parameter(torch.empty(n_head, d_k))
        nn.init.xavier_uniform_(self.K_T.data)
        self.V_T = nn.Parameter(torch.empty(n_head, n_route, d_c))
        nn.init.xavier_uniform_(self.V_T.data)

        self.Q_0 = nn.Linear(d_attribute, d_q, bias=False)
        self.Q_S = nn.Linear(n_route*d_q, d_k, bias=False)
        self.Q_T = nn.Linear(n_his*d_q, d_k, bias=False)

        self.d_q = d_q
        self.d_k = d_k
        self.d_c = d_c
        self.n_route = n_route
        self.d_out = d_out
        self.d_attribute = d_attribute
        self.w_2 = nn.Linear(d_attribute, d_out, bias=False)
        self.w_stack = nn.ModuleList([
            nn.Linear(d_attribute, d_out, bias=False) for _ in range(kt)
        ])

        self.distant_mat = dis_mat

        self.no = normal
        if self.no:
            self.norm = nn.LayerNorm(d_attribute, eps=1e-6)

    def forward(self, x):
        residual = x
        b, n, t, k = x.size()

        out = self.Q_0(x.reshape(-1, k)).reshape(b, n, t, self.d_q)

        qt = self.Q_T(out.mean(dim=1).reshape(b, t*self.d_q))
        qs = self.Q_S(out.mean(dim=2).reshape(b, n*self.d_q))

        attn_t = torch.softmax(torch.matmul(qt/self.d_k**0.5, self.K_T.transpose(0, 1)), dim=1)
        et = torch.matmul(self.V_T.permute(1, 2, 0).unsqueeze(2).unsqueeze(0), attn_t.unsqueeze(2).unsqueeze(1).unsqueeze(1)).squeeze(-1).squeeze(-1)
        attn_s = torch.softmax(torch.matmul(qs/self.d_k**0.5, self.K_S.transpose(0, 1)), dim=1)
        es = torch.matmul(self.V_S.permute(1, 2, 0).unsqueeze(2).unsqueeze(0), attn_s.unsqueeze(2).unsqueeze(1).unsqueeze(1)).squeeze(-1).squeeze(-1)

        ets = et + es
        adj = torch.matmul(ets, ets.transpose(1, 2))
        adj = self.distant_mat + adj
        adj = torch.softmax(torch.relu(adj), dim=2).unsqueeze(1)

        x = x.permute(0, 2, 1, 3)
        z = self.w_2(x.reshape(-1, k)).reshape(-1, t, n, self.d_out)
      
        for w in self.w_stack:
            z = torch.matmul(adj, z)+w(x.reshape(-1, k)).reshape(-1, t, n, self.d_out)
        z = z.permute(0, 2, 1, 3).contiguous()

        if self.no:
            z = z + residual
            z = self.norm(z)

        return z



class gcnoperation(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in,2*dim_out,bias=False)
        
    def forward(self,x,adj):
        """
        x:(b,n,k1)
        adj:(b,n,n)
        ---
        out:(b,n,k2)
        """
        out = adj@x
        out = self.linear(out)
        return out[:,:,0:self.dim_out]*torch.sigmoid(out[:,:,self.dim_out:2*self.dim_out])
 
class cascadegraphconv(nn.Module):
    def __init__(self, num_kernel, n_vertices, dim_in, dim_out):
        super().__init__()
        self.w_0 = gcnoperation(dim_in,dim_out)
        self.w_list = nn.ModuleList([gcnoperation(dim_out,dim_out)for k in range(num_kernel-1)])
        self.n_vertices = n_vertices
    def forward(self, x, adj):
        """
        x: (b,n,k1)
        adj: (b,n,n)
        ---
        out: (b,n,k2)
        """
        b,n,k = x.size()
        feature_list = []
        x = self.w_0(x,adj)
        feature_list.append(x)
        for w in self.w_list:
            x = w(x, adj)
            feature_list.append(x)
        out = torch.stack(feature_list,0)
        out,_ = torch.max(out[:,:,self.n_vertices:2*self.n_vertices,:],dim=0)
        return out       

