import torch
import torch.nn as nn
from .Layers import ConvExpandAttr, SpatioEnc, TempoEnc, AbTempoEnc, MLP, EncoderLayer, DecoderLayer


def CntLoss(model, label, pred):
    if model['name'] == 'l1':
        return torch.abs(label - pred).mean()
    elif model['name'] == 'l4':
        delta = model['delta']
        tmp = torch.abs(label - pred)
        return torch.where(tmp > delta, delta * tmp - delta * delta / 2, 0.5 * tmp**2).mean()
           

class SrcProcess(nn.Module):
    def __init__(self, opt):
        super().__init__()
        n_his = opt.n_his
        n_route, n_attr = opt.n_route, opt.n_attr

        self.CE = opt.CE['use']
        if self.CE:
            self.enc_exp = ConvExpandAttr(1, n_attr, opt.CE['kernel_size'], opt.CE['bias'])

        self.LE = opt.LE['use']
        if self.LE:
            self.enc_exp = nn.Linear(1, n_attr, bias=opt.LE['bias'])

        self.SE = opt.SE['use']
        if self.SE:
            self.enc_spa_enco = SpatioEnc(n_route, n_attr, opt.SE['no'])
        
        self.TE = opt.TE['use']
        if self.TE:
            self.enc_tem_enco = TempoEnc(n_his, n_attr, opt.TE['no'])

    def forward(self, src):
        src = self.enc_exp(src)
        
        if self.SE:
            src = self.enc_spa_enco(src)
        if self.TE:
            src = self.enc_tem_enco(src)
        
        return src


class TrgProcess(nn.Module):
    def __init__(self, opt):
        super().__init__()
        n_his, n_pred = opt.n_his, opt.n_pred
        n_route, n_attr = opt.n_route, opt.n_attr

        self.mlp = MLP(n_his, 1)

        self.CE = opt.CE['use']
        if self.CE:
            self.dec_exp = ConvExpandAttr(1, n_attr, opt.CE['kernel_size'], opt.CE['bias'])
        
        # spatio encoding
        self.SE = opt.SE['use']
        if self.SE:
            self.dec_spa_enco = SpatioEnc(n_route, n_attr, SE_spa_no = opt.SE['no'])

        # temporal encoding
        self.TE = opt.TE['use']
        if self.TE:
            self.dec_tem_enco = TempoEnc(n_pred + opt.T4N['step'], n_attr, opt.TE['no'])
        
    def forward(self, trg, enc_output, head=None, idx=None):
        head = self.mlp(enc_output)
        
        trg = self.dec_exp(trg) 
        if self.SE:
            trg = self.dec_spa_enco(trg)
        trg = torch.cat([head, trg], dim=2)
        if self.TE:
            trg = self.dec_tem_enco(trg)
        
        return trg


class Decoder(nn.Module):
    def __init__(
        self,
        opt,
        dec_slf_mask, dec_mul_mask
    ):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            DecoderLayer(opt, dec_slf_mask, dec_mul_mask)
            for _ in range(opt.n_layer)
        ])
    
    def forward(self, x, enc_output):
        for layer in self.layer_stack:
            x = layer(x, enc_output)
        return x
    

class Encoder(nn.Module):
    def __init__(
        self,
        opt,
        enc_spa_mask, enc_tem_mask
    ):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(opt, enc_spa_mask, enc_tem_mask)
            for _ in range(1)
        ])
    
    def forward(self, x):
        for layer in self.layer_stack:
            x = layer(x)
        return x
        

class STGT(nn.Module):
    def __init__(
        self,
        opt,
        enc_spa_mask, enc_tem_mask,
        dec_slf_mask, dec_mul_mask
    ):
        super().__init__()
        self.src_pro = SrcProcess(opt)
        self.trg_pro = TrgProcess(opt, self.src_pro.enc_spa_enco)

        self.dec_rdu = ConvExpandAttr(opt.n_attr, 1, opt.CE['kernel_size'], opt.CE['bias'])

        self.encoder = Encoder(opt, enc_spa_mask, enc_tem_mask)
        self.decoder = Decoder(opt, dec_slf_mask, dec_mul_mask)

        self.T4N = opt.T4N['use']
        if self.T4N:
            self.T4N_step = opt.T4N['step']
            self.change_head = opt.T4N['change_head']
            self.change_enc = opt.T4N['change_enc']
            self.T4N_end = opt.T4N['end_epoch']
        
        self.loss_fn = opt.loss_fn

        self.n_pred = opt.n_pred
    
    def forward(self, src, label, epoch=1e8):
        enc_input = self.src_pro(src)
        enc_output = self.encoder(enc_input)
        enc_output_4head = enc_output

        trg = label[:, :, :self.n_pred, 0].unsqueeze(-1)
        loss = 0.0
        dec_output = None

        if self.T4N and epoch < self.T4N_end:
            for i in range(self.T4N_step):
                dec_input = self.trg_pro(trg, enc_output_4head)
                dec_output = self.decoder(dec_input, enc_output)

                if self.change_head and i < self.T4N_step - 1:
                    pre = enc_output[:, :, 1:, :]
                    post = dec_output[:, :, 0, :].unsqueeze(2)
                    enc_output_4head = torch.cat([pre, post], dim=2)
                
                if self.change_enc:
                    enc_output = enc_output_4head
                
                dec_output = self.dec_rdu(dec_output)
                trg = dec_output[:, :, 1:, :]

                loss = loss + CntLoss(self.loss_fn, label[:, :, i:i+self.n_pred, :], dec_output[:, :, :-1, :])
            return dec_output[:, :, :-1, :], loss