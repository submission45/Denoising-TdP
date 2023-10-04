import os.path

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models.tdp.utils_tdp import compute_logits_return_labels_and_predictions

from models.tdp.utils_tdp import load_model_tdp
from utils.ml_utils import compute_accuracy

def evaluation_tdp_classifer(device=0, partition: str='holdout', d_type: str='original'):

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    x = torch.Tensor(np.load(f"experiment_data/{d_type}/hb_{partition}.npy"))

    if len(x.shape) > 2:
        x = x.squeeze()
    y = torch.Tensor(np.load(f"experiment_data/hb_labels_{partition}.npy"))

    filename_pred = f'experiment_data/{d_type}/hb_prediction_{partition}.npy'
    filename_soft = f'experiment_data/{d_type}/hb_probability_{partition}.npy'

    if os.path.exists(filename_pred):
        hard_predictions = torch.Tensor(np.load(filename_pred))
        labels = y

    else:

        test_dataset = TensorDataset(x, y)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        args_tdp = {
            'checkpoints_dir': '.',
            'device': device,
            'model_dir': f'tdp_{d_type}'
        }

        tdp_model = load_model_tdp(**args_tdp)
        tdp_model.eval()
        logits, labels, hard_predictions, soft_predictions = compute_logits_return_labels_and_predictions(tdp_model, test_dataloader, device)
        print(soft_predictions.shape)

        if not os.path.exists(filename_pred):
            np.save(filename_pred, hard_predictions)
        if not os.path.exists(filename_soft):
            np.save(filename_soft, soft_predictions)

    acc = compute_accuracy(hard_predictions, labels)

    print("Accuracy of", d_type, acc)

    return acc


if __name__ == '__main__':
    evaluation_tdp_classifer(device='cpu', d_type='original')
