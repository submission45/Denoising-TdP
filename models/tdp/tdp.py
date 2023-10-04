import json

import torch
import torch.nn as nn
import pandas as pd

from .dense_net import DenseTrunk
from .layers import Conv1DBlock, Activation

class TdPModel(nn.Module):
    def __init__(self, input_channels=1, layers=0, kernels=3, growth_rate=12,  dropout_rate=None, bottleneck=False, compression=1.0,
                 depth=40, activation=None, conv_padding=None, conv_strides=1, conv_dilation=1,
                 grow_layers_channels=True, pool_steps=[2], verbose=True, encoder_pool_type="avg", n_classes=2):
        super(TdPModel, self).__init__()

        if verbose:
            print("====================================================")
            print("                      TdP Model                     ")
            print("====================================================")

        assert type(pool_steps) in (list, tuple) and len(pool_steps) > 0, "Bad pool steps"

        transition_block_kwargs = []
        # Pooling: max or avg
        for step in pool_steps:
            transition_block_kwargs.append({"pool_type": encoder_pool_type, "pool_size": step, "strides": step})
        # Convolution dimension reduction: strides
        # for step in pool_steps:
        #     transition_block_kwargs.append({"pool_type": "convolution", "strides": step})
        blocks = len(transition_block_kwargs) + 1

        if activation is None:
            activation = {'name': "relu"}

        self.__dense_trunk = DenseTrunk(input_channels=input_channels, blocks=blocks, layers=layers, kernels=kernels,
                                        growth_rate=growth_rate,
                                        dropout_rate=dropout_rate, bottleneck=bottleneck, compression=compression,
                                        depth=depth, activation=activation, conv_padding=conv_padding,
                                        conv_strides=conv_strides, conv_dilation=conv_dilation,
                                        grow_layers_channels=grow_layers_channels,
                                        transition_block_kwargs=transition_block_kwargs, verbose=verbose)

        out_channels = self.__dense_trunk.get_output_channels()
        self.__transition = nn.AdaptiveAvgPool1d(1)
        self.__classifier = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=1024),
            Activation(**activation),
            nn.Linear(in_features=1024, out_features=512),
            Activation(**activation),
            nn.Linear(in_features=512, out_features=64),
            Activation(**activation),
            nn.Linear(in_features=64, out_features=20),
            Activation(**activation),
            nn.Dropout(0.2),
            nn.Linear(in_features=20, out_features=n_classes),
            # nn.Sigmoid()
            # nn.Linear(in_features=20, out_features=2),
            # nn.LogSoftmax(dim=1)
        )

        if verbose:
            print(f"Transition layers:")
            print(pd.DataFrame(transition_block_kwargs))

        if verbose:
            print("====================================================")

    def forward(self, x):
        features = self.__dense_trunk(x)
        logits = self.__transition(features)
        return self.__classifier(torch.flatten(logits, start_dim=1))