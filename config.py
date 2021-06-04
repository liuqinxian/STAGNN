import warnings
import os.path
import sys
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.utils import weight_matrix


class DefaultConfig(object):
    seed = 666
    device = 0

    scaler = StandardScaler()
    day_slot = 288
    n_route, n_his, n_pred = 228, 12, 12
    n_train, n_val, n_test = 34, 5, 5
    
    mode = 1
    # 1: 3, 6, 9, 12
    # 2: 3, 6, 12, 18, 24

    model = 'STAGNN'
    name = 'PeMS'
    batch_size = 50
    lr = 1e-3
    
    adam = {'use': True, 'weight_decay': 1e-4}
    slr = {'use': True, 'step_size': 300, 'gamma': 0.3}
    
    resume = False
    start_epoch = 0
    epochs = 1500
    
    n_layer = 1
    n_attr, n_hid = 33, 512
    drop_prob = 0.0
    
    # expand attr by conv
    CE = {'use': True, 'kernel_size': 1, 'bias': False}

    # spatio encoding
    SE = {'use': True, 'separate': True, 'no': False}
    # tempo encoding
    TE = {'use': True, 'no': True}
    
    # MultiHeadAttention
    attn = {'head': 1, 'd_k': 33, 'd_v': 33, 'drop_prob': 0}
   
    # STAttnGraphConv
    ST = {'use': True, 'n_head': 4, 'd_q': 32, 'd_k': 128, 'd_c': 10, 'kt': 3, 'normal': False}
    
    # TeaforN
    T4N = {'use': True, 'step': 2, 'end_epoch': 10000, 'change_head': True, 'change_enc': True}

    data_path = 'PeMS/V_228.csv'
    adj_matrix_path = 'PeMS/W_228.csv'
    dis_mat = {'name': 'L', 'matrix': 0.0}

    prefix = 'log/' + name + '/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    checkpoint_temp_path = prefix + '/temp.pth'
    checkpoint_best_path = prefix + '/best.pth'
    tensorboard_path = prefix
    record_path = prefix + 'record.txt'
    
    eps = 0.1
    if dis_mat['name'] == 'A':
        dis_mat['matrix'] = pd.read_csv(adj_matrix_path, header=None).values
        dis_mat['matrix'] = torch.from_numpy(dis_mat['matrix']).float()
    elif dis_mat['name'] == 'L':
        dis_mat['matrix'] = weight_matrix(adj_matrix_path, epsilon=eps)
        dis_mat['matrix'] = torch.from_numpy(dis_mat['matrix']).float()
    elif dis_mat['name'] == 'N':
        dis_mat['matrix'] = 0.0


    def parse(self, kwargs):
        '''
        customize configuration by input in terminal
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has no attribute %s' % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


class Logger(object):
    def __init__(self, file_name='Default.log'):

        self.terminal = sys.stdout
        self.log = open(file_name, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    