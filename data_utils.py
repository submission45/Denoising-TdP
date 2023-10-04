import numpy as np

def normalize_signals_for_comparison(signal_1_d):
    return (signal_1_d - signal_1_d.min()) / (signal_1_d.max() - signal_1_d.min())


def filter_noise_data(data, threshold=2):

    def has_consecutive_constant(row, threshold):
        consecutive_count = 1
        max_consecutive_count = 1
        for i in range(1, len(row)):
            if row[i] == row[i - 1]:
                consecutive_count += 1
                max_consecutive_count = max(max_consecutive_count, consecutive_count)
            else:
                consecutive_count = 1
        return max_consecutive_count > threshold  # len(row) / 2

    mask = np.array([not has_consecutive_constant(row, threshold) for row in data])
    return mask

def load_array_from_name(denoiser_name):
    signal = np.load(f'experiment_data/noised/hb_holdout_BW_{denoiser_name}.npy').squeeze()

    if signal.shape[1] != 500:
        import neurokit2 as nk
        signal = np.apply_along_axis(nk.signal_resample, axis=1, arr=signal, method="FFT", desired_length=500)
    return signal