import numpy as np

from utils.ml_utils import ssd, mad, prd, cosine_sim, snr
from utils.data_utils import normalize_signals_for_comparison, filter_noise_data, load_array_from_name

# Metric values
def compute_all_metrics(y, y_pred):
    ssd_ = ssd(y, y_pred)
    mad_ = mad(y, y_pred)
    prd_ = prd(y, y_pred)
    snr_ = snr(y, y_pred)
    cosine_sim_ = cosine_sim(y, y_pred)
    metrics = [ssd_, mad_, prd_, snr_, cosine_sim_]
    dict_metrics = {
        'SSD': rf"{'{:.2f}'.format(ssd_.mean())}+{'{:.2f}'.format(ssd_.std())}",
        'MAD': rf"{'{:.2f}'.format(mad_.mean())}+{'{:.2f}'.format(mad_.std())}",
        'PRDN': rf"{'{:.2f}'.format(prd_.mean())}+{'{:.2f}'.format(prd_.std())}",
        'SNR': rf"{'{:.2f}'.format(snr_.mean())}+{'{:.2f}'.format(snr_.std())}",
        'CosS': rf"{'{:.2f}'.format(cosine_sim_.mean())}+{'{:.2f}'.format(cosine_sim_.std())}",
    }
    return metrics, dict_metrics

def evaluation_denoising():
    denoisers_name = ['noised', 'drnn', 'descod', 'deepfilter', 'fcn_dae', 'wavelet']
    denoisers_x = {}

    for den_name in denoisers_name:
        denoisers_x[den_name] = np.apply_along_axis(func1d=normalize_signals_for_comparison, arr=load_array_from_name(denoiser_name=den_name), axis=1)

    noise = denoisers_x['noised']
    index_to_save = filter_noise_data(noise)
    for k, v in denoisers_x.items():
        denoisers_x[k] = v[index_to_save]
    denoisers_x['original'] = np.load('experiment_data/original/hb_holdout.npy')[index_to_save]

    results = {'denoiser': [], 'metric': [], 'value': []}
    original_signal = denoisers_x['original']
    metrics_name = ['SSD', 'MAD', 'PRDN', 'SNR', 'CosS']

    for d_name, values in denoisers_x.items():
        metrics, mean_metrics = compute_all_metrics(original_signal, values)
        print(d_name, mean_metrics)

        for metric_index in range(len(metrics_name)):
            r_list = metrics[metric_index]
            results['denoiser'] += [d_name] * len(r_list)
            results['value'] += list(r_list)
            results['metric'] += [metrics_name[metric_index]] * len(r_list)
    return results

if __name__ == '__main__':
    evaluation_denoising()