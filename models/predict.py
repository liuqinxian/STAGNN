import torch


def predict(model, src, trg, opt):
    '''
    auto regression
    '''
    model.eval()
    with torch.no_grad():
        enc_input = model.src_pro(src)
        enc_output = model.encoder(enc_input)

        lines = trg.shape[2]
        
        trg = torch.zeros(trg.shape).cuda()

        for i in range(lines):
            dec_input = model.trg_pro(trg, enc_output)
            dec_output = model.decoder(dec_input, enc_output)
            dec_output = model.dec_rdu(dec_output)
            trg[:, :, i, :] = dec_output[:, :, i, :]

        return trg