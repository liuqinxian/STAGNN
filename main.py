import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import models
import sys
import os
from models import predict
from data import STAGNN_Dataset
from torch.utils.tensorboard import SummaryWriter
from utils.utils import evaluate_metric
from config import DefaultConfig, Logger


opt = DefaultConfig()

sys.stdout = Logger(opt.record_path)

# random seed
seed = opt.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)


def test(model, loss_fn, test_iter, opt):
    model.eval()
    loss_sum, n = 0.0, 0
    for x, y in test_iter:
        y_pred = predict(model, x, y, opt)
        if opt.AT['use']:
            y = y[:, :, :, 0].unsqueeze(-1)
        loss = loss_fn(y_pred, y)
        loss_sum += loss.item()
        n += 1
    return loss_sum / n


def train(**kwargs):
    opt.parse(kwargs)
    
    # load data
    batch_size = opt.batch_size
    train_dataset = STAGNN_Dataset(opt, train=True, val=False)
    val_dataset = STAGNN_Dataset(opt, train=False, val=True)
    test_dataset = STAGNN_Dataset(opt, train=False, val=False)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size)
    
    # mask
    n_route = opt.n_route
    n_his = opt.n_his
    n_pred = opt.n_pred
    enc_spa_mask = torch.ones(1, 1, n_route, n_route).cuda()
    enc_tem_mask = torch.ones(1, 1, n_his, n_his).cuda()
    dec_slf_mask = torch.tril(torch.ones((1, 1, n_pred + 1, n_pred + 1)), diagonal=0).cuda()
    dec_mul_mask = torch.ones(1, 1, n_pred + 1, n_his).cuda()
    
    # loss
    loss_fn = nn.L1Loss()
    
    # model
    model = getattr(models, opt.model)(
        opt,
        enc_spa_mask, enc_tem_mask,
        dec_slf_mask, dec_mul_mask
    )
    model.cuda()
    
    # optimizer
    lr = opt.lr
    if opt.adam['use']:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = opt.adam['weight_decay'])
    
    # scheduler
    if opt.slr['use']:
        step_size, gamma = opt.slr['step_size'], opt.slr['gamma']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif opt.mslr['use']:
        milestones, gamma = opt.mslr['milestones'], opt.mslr['gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    
    # resume
    start_epoch = opt.start_epoch
    min_val_loss = np.inf
    checkpoint_temp_path = opt.checkpoint_temp_path
    if opt.resume:
        if os.path.isfile(checkpoint_temp_path):
            checkpoint = torch.load(checkpoint_temp_path)
            start_epoch = checkpoint['epoch'] + 1
            min_val_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint (epoch {})'.format(checkpoint['epoch']))  
            
    # tensorboard
    tensorboard_path = opt.tensorboard_path + str(start_epoch)
    writer = SummaryWriter(tensorboard_path)
    
    # train
    name = opt.name
    epochs = opt.epochs
    checkpoint = None
    checkpoint_temp_path = opt.checkpoint_temp_path
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()        
        loss_sum, n = 0.0, 0
        for x, y in train_iter:
            _, loss = model(x, y, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n += 1
        scheduler.step()
        
        model.eval()
        
        val_loss = test(model, loss_fn, val_iter, opt)
        print('epoch', epoch, ' ', name, ', train loss:', loss_sum / n, ', validation loss:', val_loss, ', lr:', optimizer.param_groups[0]['lr'])

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'min_loss': min_val_loss,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_temp_path)

        writer.add_scalar('train loss', loss_sum / n, epoch)
        writer.add_scalar('test loss', val_loss, epoch)

    checkpoint_best_path = opt.checkpoint_best_path
    torch.save(checkpoint, checkpoint_best_path)
    
    checkpoint = torch.load(checkpoint_best_path)
    best_model = getattr(models, opt.model)(
        opt,
        enc_spa_mask, enc_tem_mask,
        dec_slf_mask, dec_mul_mask
    )
    best_model.load_state_dict(checkpoint['model'])
    best_model.cuda()
    best_model.eval()
    
    writer = SummaryWriter('/tmp')
    
    test_loss = test(best_model, loss_fn, test_iter, opt)
    if opt.mode == 1:
        MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, opt)
        print("test loss:", test_loss, "\nMAE:", MAE, ", MAPE:", MAPE, "%, RMSE:", RMSE)
    elif opt.mode == 2:
        RAE, RSE, COR = evaluate_metric(best_model, test_iter, opt)
        print("test loss:", test_loss, "\nRAE:", RAE, ", RSE:", RSE, "%, RMSE:", COR)
    print('='*20)


if __name__ == '__main__':
    import fire
    fire.Fire()