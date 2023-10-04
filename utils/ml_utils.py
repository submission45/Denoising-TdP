import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_accuracy(predictions: torch.tensor, targets: torch.tensor):
    """
    compute the model's accuracy
    :param predictions: tensor containing the predicted labels
    :param targets: tensor containing the target labels
    :return: the accuracy in [0, 1]
    """
    accuracy = torch.div(torch.sum(predictions == targets), len(targets))
    return accuracy

def ssd(y, y_pred):
    return np.sum((y - y_pred) ** 2, axis=1)  # axis 1 is the signal dimension

def mad(y, y_pred):
    return np.max(np.abs(y - y_pred), axis=1)  # axis 1 is the signal dimension

def prd(y, y_pred, mean=True):
    num = ssd(y, y_pred)
    if mean:
        den = np.sum((y - np.mean(y, axis=1, keepdims=True)) ** 2, axis=1)
    else:
        den = np.sum(y ** 2, axis=1)
    den[den == 0] = 1e-8
    prd = np.sqrt(num / den) * 100
    return prd

def snr(y, y_pred):
    snr = 10 * np.log10(((prd(y, y_pred, mean=False) ** -1) / 100))
    return snr

def cosine_sim(y, y_pred):
    y = np.nan_to_num(y)
    y_pred = np.nan_to_num(y_pred)
    cos_sim = []
    for idx in range(len(y)):
        kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
        cos_sim.append(kl_temp)
    cos_sim = np.array(cos_sim)
    return cos_sim.squeeze()

def snr_imp(y_in, y_out, y_clean):
    return snr(y_clean, y_out) - snr(y_clean, y_in)
