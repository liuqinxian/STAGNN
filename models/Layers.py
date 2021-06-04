import torch
import torch.nn as nn
from .SubLayers import STAttnGraphConv
from .TransformerLayers import MultiHeadAttention, PositionwiseFeedForward


class ConvExpandAttr(nn.Module):
    '''
    [batch, n_route, n_time, 1] -> [batch, n_route, n_time, n_attr]
    '''
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size,
        bias
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, bias=bias)
    
    def forward(self, x):
        # [batch, n_route, n_time, 1] -> [batch, 1, n_route, n_time]
        x = x.permute(0, 3, 1, 2)
        # [batch, 1, n_route, n_time] -> [batch, n_attr, n_route, n_time]
        x = self.conv(x)
        # [batch, n_attr, n_route, n_time] -> [batch, n_route, n_time, n_attr]
        x = x.permute(0, 2, 3, 1)
        return x
    

class SpatioEnc(nn.Module):
    def __init__(
        self,
        n_route,
        n_attr=33,
        normal=True
    ):
        super().__init__()
        
        self.enc = nn.Parameter(torch.empty(n_route, n_route))
        self.w = nn.Linear(n_route, n_attr)
        self.no = normal
        self.norm = nn.LayerNorm(n_attr, eps=1e-6)
        
        nn.init.xavier_uniform_(self.enc.data)
    
    def forward(self, x):
        enc = self.w(self.enc)
        x = x.permute(0, 2, 1, 3) + enc
        if self.no:
            x = self.norm(x)
        x = x.permute(0, 2, 1, 3)
        return x
    

class TempoEnc(nn.Module):
    def __init__(
        self,
        n_time,
        n_attr,
        normal=True
    ):
        super().__init__()
        
        self.time = n_time
        self.enc = nn.Embedding(n_time, n_attr)
        self.no = normal
        self.norm = nn.LayerNorm(n_attr, eps=1e-6)
        
    def forward(self, x, start=0):
        length = x.shape[2]
        enc = self.enc(torch.arange(start, start + length).cuda())
        x = x + enc
        if self.no:
            x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=1
    ):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_in, d_in//2),
            nn.ReLU(inplace=True),
            nn.Linear(d_in//2, d_in//4),
            nn.ReLU(inplace=True),
            nn.Linear(d_in//4, d_out)
        )

    def forward(self, x):
        # [batch, n_route, n_his, n_attr] -> [batch, n_route, n_attr, n_his]
        x = x.permute(0, 1, 3, 2)
        # [batch, n_route, n_attr, n_his] -> [batch ,n_route, n_attr, 1]
        output = self.linear(x)
        # [batch ,n_route, n_attr, 1] -> [batch, n_route, 1, n_attr]
        output = output.permute(0, 1, 3, 2)
        return output


class EncoderLayer(nn.Module):
    def __init__(
        self,
        opt,
        spa_mask, tem_mask
    ):
        super().__init__()
        
        n_route, n_his, n_attr, n_hid = opt.n_route, opt.n_his, opt.n_attr, opt.n_hid

        dis_mat = opt.dis_mat
        
        self.tem_attn = MultiHeadAttention(opt.attn['head'], n_attr, opt.attn['d_k'], opt.attn['d_v'], opt.attn['drop_prob'])
        self.tem_mask = tem_mask

            
        self.ST = opt.ST['use']
        if self.ST:
            n_head, d_q, d_k, d_c, kt, normal = opt.ST['n_head'], opt.ST['d_q'], opt.ST['d_k'], opt.ST['d_c'], opt.ST['kt'], opt.ST['normal']
            self.stgc = STAttnGraphConv(n_route, n_his, n_attr, n_attr, dis_mat, n_head, d_q, d_k, d_c, kt, normal)
        
        self.pos_ff = PositionwiseFeedForward(n_attr, n_hid, opt.drop_prob)
        
    def forward(self, x):
        x = self.tem_attn(x, x, x, self.tem_mask)
            
        if self.ST:
            x = self.stgc(x)
        
        x = self.pos_ff(x)
        return x
        

class DecoderLayer(nn.Module):
    def __init__(
        self,
        opt,
        slf_mask, mul_mask
    ):
        super().__init__()
        
        n_attr, n_hid = opt.n_attr, opt.n_hid
   
        self.slf_attn = MultiHeadAttention(opt.attn['head'], n_attr, opt.attn['d_k'], opt.attn['d_v'], opt.attn['drop_prob'])
        self.slf_mask = slf_mask

        self.mul_attn = MultiHeadAttention(opt.attn['head'], n_attr, opt.attn['d_k'], opt.attn['d_v'], opt.attn['drop_prob'])
        self.mul_mask = mul_mask
        
        self.pos_ff = PositionwiseFeedForward(n_attr, n_hid, opt.drop_prob)

    def forward(self, x, enc_output):
        x = self.mul_attn(x, enc_output, enc_output, self.mul_mask)
        x = self.pos_ff(x)
        return x
        
        
        
        
        