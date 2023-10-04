import os
import torch
import numpy as np

from models.tdp.train_tdp import train_model
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def main_train(device='cpu', batch_size: int=64, d_type: str='tdp_original'):
    assert d_type in ['original', 'drnn', 'descod', 'deepfilter'], "Available denoisers are 'original', 'drnn', 'descod', 'deepfilter'"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    filename_train = f"experiment_data/{d_type}/hb_training.npy"
    filename_val = f"experiment_data/{d_type}/hb_validation.npy"

    print(filename_train)
    print(filename_val)

    x = torch.Tensor(np.load(filename_train))
    if len(x.shape) > 2:
        x = x.squeeze()
    y = torch.Tensor(np.load(f"experiment_data/hb_labels_training.npy"))

    train_dataset = TensorDataset(x, y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    x_val = torch.Tensor(np.load(filename_val))
    if len(x_val.shape) > 2:
        x_val = x_val.squeeze()
    y_val = torch.Tensor(np.load(f"experiment_data/hb_labels_validation.npy"))

    valid_dataset = TensorDataset(x_val, y_val)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    dirname = f"checkpoints/tdp_{d_type}"
    print(dirname)
    os.makedirs(dirname, exist_ok=True)

    model = train_model(train_dataloader, valid_dataloader, device, dirname, epochs=100)

    return model

if __name__ == '__main__':
    main_train(d_type='original')
