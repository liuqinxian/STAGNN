import torch
import numpy as np
import pandas as pd
from models import predict


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels.abs()>1e-6)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels.abs()>1e-6)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels.abs()>1e-6)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss


def evaluate_metric(model, data_iter, opt):
    model.eval()
    scaler = opt.scaler
    n_pred = opt.n_pred
    
    length = n_pred // 3
    with torch.no_grad():
        mae = [[] for _ in range(length)]
        mape = [[] for _ in range(length)]
        mse = [[] for _ in range(length)]
        MAE, MAPE, RMSE = [0.0] * length, [0.0] * length, [0.0] * length

        for x, y in data_iter:            
            y_pred = predict(model, x, y, opt).permute(0, 3, 2, 1)
            y_pred = scaler.inverse_transform(y_pred.cpu().numpy())
          
            for i in range(length):
                y_pred_select = y_pred[:, :, 3 * i + 2, :].reshape(-1)
                y_select = y[:, :, 3 * i + 2, :].reshape(-1)
                d = np.abs(y_select - y_pred_select)

                y_pred_select = torch.from_numpy(y_pred_select)
                y_select = torch.from_numpy(y_select)
                mae[i] += masked_mae(y_pred_select, y_select,0.0).numpy().tolist()
                mape[i] += masked_mape(y_pred_select, y_select,0.0).numpy().tolist()
                mse[i] += masked_mse(y_pred_select, y_select,0.0).numpy().tolist()

        for j in range(length):
            MAE[j] = np.array(mae[j]).mean()
            MAPE[j] = 100.0 * (np.array(mape[j]).mean())
            RMSE[j] = np.sqrt(np.array(mse[j]).mean())
        
        return MAE, MAPE, RMSE
    
    
def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W
