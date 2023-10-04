# The Negative Impact of Denoising on Automated Classification of Electrocardiograms

### Current package structure
The training code for the denoiser models is available at <a href="https://github.com/fperdigon/DeepFilter/tree/master">this</a> link. 
```
Package
Denoising+TdP/
.
├── README.md
├── checkpoints
│   ├── tdp_deepfilter
│   │   └── model_best.pt
│   ├── tdp_descod
│   │   └── model_best.pt
│   ├── tdp_drnn
│   │   └── model_best.pt
│   └── tdp_original
│       └── model_best.pt
├── environment.yml
├── experiment_data
│   ├── deepfilter
│   │   ├── hb_holdout.npy
│   │   ├── hb_training.npy
│   │   └── hb_validation.npy
│   ├── descod
│   │   ├── hb_holdout.npy
│   │   ├── hb_training.npy
│   │   └── hb_validation.npy
│   ├── drnn
│   │   ├── hb_holdout.npy
│   │   ├── hb_training.npy
│   │   └── hb_validation.npy
│   ├── hb_labels_holdout.npy
│   ├── hb_labels_training.npy
│   ├── hb_labels_validation.npy
│   ├── noised
│   │   ├── hb_holdout_BW_deepfilter.npy
│   │   ├── hb_holdout_BW_descod.npy
│   │   ├── hb_holdout_BW_drnn.npy
│   │   ├── hb_holdout_BW_fcn_dae.npy
│   │   ├── hb_holdout_BW_original.npy
│   │   └── hb_holdout_BW_wavelet.npy
│   └── original
│       ├── hb_holdout.npy
│       ├── hb_training.npy
│       └── hb_validation.npy
├── main.py
├── models
│   ├── __init__.py
│   └── tdp
│       ├── __init__.py
│       ├── nn
│       │   ├── __init__.py
│       │   ├── dense_net.py
│       │   ├── layers.py
│       │   └── tdp.py
│       ├── train_tdp.py
│       └── utils_tdp.py
├── pipeline_evaluation_classification.py
├── pipeline_evaluation_denoising.py
├── pipeline_train_tdp.py
└── utils
    ├── __init__.py
    ├── data_utils.py
    └── ml_utils.py
```

#### Usage

To execute:
- Create and install the environment:
```console
foo@bar:~$ conda env create -f environment.yml
```
- Activate the environment:
```console
foo@bar:~$ source activate denoising_tdp
```
- Launch the test from CLI:
```console
(denoising_tdp) foo@bar:~$ python pipeline_evaluation_classification.py 
(denoising_tdp) foo@bar:~$ python pipeline_evaluation_denoising.py
(denoising_tdp) foo@bar:~$ python pipeline_train_tdp.py
```
#### Please note
The Generepol dataset is not public. A small portion of it nevertheless added to the repo for reproducibility.






