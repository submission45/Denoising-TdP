import torch
import torch.nn as nn

from .layers import Conv1DBlock, Activation


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=None, compression=1.0, activation=None, padding=0, strides=2, dilation=1,
                 pool_type="avg", pool_size=2, convolution_kernel=3, transposition_channels=1, transposition_kernel=3, transposition_output_padding=None,
                 enable_compression_block=True):
        super(TransitionBlock, self).__init__()

        self.pool_type = pool_type
        self.enable_compression_block = enable_compression_block
        self.dropout_rate = dropout_rate

        if activation is None:
            activation = {"name": "leaky_relu", "args": {"negative_slope": 0.1}}

        if enable_compression_block:
            self.compression_block = Conv1DBlock(in_channels=in_channels,
                                                 out_channels=int(out_channels * compression), kernel=1,
                                                 activation=activation, padding=padding, order="conv_bn_act")
            self.__final_output_channels = int(out_channels * compression)
            in_channels = self.__final_output_channels
        else:
            self.__final_output_channels = out_channels
        if dropout_rate:
            self.dropout_layer = nn.Dropout1d(p=dropout_rate)

        if pool_type == "avg":
            self.pooling_layer = nn.AvgPool1d(kernel_size=pool_size, stride=strides)
        elif pool_type == "max":
            self.pooling_layer = nn.MaxPool1d(kernel_size=pool_size, stride=strides)
        elif pool_type == "convolution":
            # We set out_channels=in_channels so we do not alter the number of channels but only perform a dimension reduction using the strides
            conv_padding = convolution_kernel // 2
            self.pooling_layer = Conv1DBlock(in_channels=in_channels, out_channels=in_channels,
                                             kernel=convolution_kernel, activation=activation, padding=conv_padding,
                                             strides=strides, order="conv_bn_act")
        elif pool_type == "conv_transpose":
            if transposition_output_padding is None:
                transposition_output_padding = strides-1
            transpo_padding = transposition_kernel // 2
            transposition_channels = 1
            self.deconvolution_layer = nn.Sequential(
                nn.ConvTranspose1d(in_channels=in_channels, out_channels=transposition_channels,
                                   kernel_size=transposition_kernel, padding=transpo_padding, stride=strides,
                                   output_padding=transposition_output_padding),
                nn.BatchNorm1d(transposition_channels),
                Activation(**activation)
            )
            self.__final_output_channels = 1
        else:
            raise ValueError("Invalid pool_type")

    def get_output_channels(self):
        return self.__final_output_channels

    def forward(self, x):
        logits = x
        if self.enable_compression_block:
            logits = self.compression_block(logits)
        if self.dropout_rate:
            logits = self.dropout_layer(logits)
        if self.pool_type in ("avg", "max", "convolution"):
            logits = self.pooling_layer(logits)
        elif self.pool_type == "conv_transpose":
            logits = self.deconvolution_layer(logits)
        return logits

class DenseBlock(nn.Module):
    def __init__(self, layers, in_channels, out_channels, kernel, growth_rate, dropout_rate=None, bottleneck=False, activation=None, grow_channels=True):
        super(DenseBlock, self).__init__()

        if activation is None:
            activation = {"name": "leaky_relu", "args": {"negative_slope": 0.1}}

        self.__layers = []
        for i in range(layers):
            layer = Conv1DBlock(in_channels=in_channels, out_channels=out_channels, kernel=kernel,
                                dropout_rate=dropout_rate, bottleneck=bottleneck, order="conv_bn_act",
                                activation=activation, padding=kernel // 2, strides=1, dilation=1)
            self.add_module(f"DenseLayer_{i}", layer)
            self.__layers.append(layer)
            in_channels += out_channels
            if grow_channels:
                out_channels += growth_rate

        # Set DenseBlock final output number of channels
        self.__final_output_channels = in_channels

    def get_output_channels(self):
        return self.__final_output_channels

    def forward(self, x):
        logits = x
        feature_maps = [logits]
        for layer in self.__layers:
            feature_maps.append(layer(logits))
            logits = torch.cat(feature_maps, dim=1)
        return logits

class DenseTrunk(nn.Module):
    def __init__(self, input_channels, blocks=3, layers=0, kernels=3, growth_rate=12, dropout_rate=None, bottleneck=False,
                 compression=1.0, depth=40, activation=None, conv_padding=None, conv_strides=1, conv_dilation=1,
                 grow_layers_channels=True, transition_block_kwargs=None, verbose=True):
        super(DenseTrunk, self).__init__()

        assert compression > 0.0 and compression <= 1.0, "Compression must be between 0.0 and 1.0. Setting compression to 1.0 will turn it off"

        if type(layers) is list:
            assert len(layers) == blocks, "Number of blocks have to be same to specified layers"
        elif layers == 0:
            if bottleneck:
                layers = (depth - (blocks + 1)) / blocks // 2
            else:
                layers = (depth - (blocks + 1)) // blocks
            layers = [int(layers) for _ in range(blocks)]
        else:
            layers = [int(layers) for _ in range(blocks)]

        if type(kernels) in (list, tuple):
            assert len(kernels) == blocks, "Number of dense blocks have to be same length to specified kernels"
        else:
            kernels = [int(kernels) for _ in range(blocks)]

        if transition_block_kwargs is not None:
            assert type(transition_block_kwargs) in (list, tuple) and len(transition_block_kwargs) == blocks-1, "Transition blocks arguments must be a list or tuple of length : blocks - 1"
            use_transition_kwargs = True
        else:
            use_transition_kwargs = False

        filters = growth_rate * 2
        initial_kernel = 3

        if activation is None:
            activation = {'name': "relu"}

        if verbose:
            print(f"Blocks: {blocks}")
            print(f"Layers per block: {layers}")

        if conv_padding is None:
            conv_padding = initial_kernel//2
        self.__initial_layer = Conv1DBlock(in_channels=input_channels, out_channels=filters, kernel=initial_kernel,
                                           activation=activation, padding=conv_padding, strides=conv_strides,
                                           dilation=conv_dilation, order="conv_bn_act")

        self.__blocks = []
        future_in_channels = filters
        for block_number in range(blocks):
            block = DenseBlock(layers=layers[block_number], in_channels=future_in_channels, out_channels=filters,
                               kernel=kernels[block_number], growth_rate=growth_rate,
                               grow_channels=grow_layers_channels, bottleneck=bottleneck, activation=activation)
            self.add_module(f"DenseBlock_{block_number}", block)
            self.__blocks.append(block)
            future_in_channels = block.get_output_channels()
            filters = growth_rate * layers[block_number]
            if block_number < blocks - 1:
                transition_block = TransitionBlock(in_channels=future_in_channels, out_channels=filters, dropout_rate=dropout_rate,
                                compression=compression, activation=activation, transposition_channels=filters,
                                **transition_block_kwargs[block_number] if use_transition_kwargs else dict())
                self.add_module(f"TransitionBlock_{block_number}", transition_block)
                self.__blocks.append(transition_block)
                future_in_channels = transition_block.get_output_channels()
                filters = int(filters * compression)
        self.__final_output_channels = future_in_channels

    def get_output_channels(self):
        return self.__final_output_channels

    def forward(self, x):
        logits = self.__initial_layer(x)

        for block in self.__blocks:
            logits = block(logits)
        return logits