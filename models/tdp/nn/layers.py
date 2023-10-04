import torch.nn as nn

class Activation(nn.Module):
    __CLASS_ACTIVATIONS = {
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "softmax": nn.Softmax,
        "sigmoid": nn.Sigmoid
    }

    def __init__(self, desc=None, name=None, args=None, **kwargs):
        super(Activation, self).__init__()

        if desc is not None:
            self.configure(desc['name'], desc['args'] if 'args' in desc else None)
        else:
            self.configure(name, args)

    def configure(self, name, args):
        if name in self.__CLASS_ACTIVATIONS:
            if type(args) is tuple or type(args) is list:
                self.__activation = self.__CLASS_ACTIVATIONS[name](*args)
            elif type(args) is dict:
                self.__activation = self.__CLASS_ACTIVATIONS[name](**args)
            else:
                raise ValueError("Unexpected value for args")
        else:
            raise ValueError("Unknown activation")

    def forward(self, x):
        return self.__activation(x)

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dropout_rate=None, bottleneck=False, bottleneck_width=4, activation=None,
                 dilation=1, padding=0, strides=1, order="bn_act_conv"):
        super(Conv1DBlock, self).__init__()

        self.bottleneck = bottleneck

        if activation is None:
            activation = {"name": "leaky_relu", "args": {"negative_slope": 0.1}}

        if bottleneck:
            if self.bottleneck == "before":
                bottleneck_in_channels = in_channels
                # Update in_channels with last output of bottleneck layer if enabled
                in_channels = out_channels * bottleneck_width
            elif self.bottleneck == "after":
                bottleneck_in_channels = out_channels

            if order == "bn_act_conv":
                self._bottleneck_layers = nn.Sequential(
                    nn.BatchNorm1d(in_channels),
                    Activation(**activation),
                    nn.Conv1d(in_channels=in_channels, out_channels=out_channels * bottleneck_width, kernel_size=1, bias=True, stride=strides))
            elif order == "conv_bn_act":
                self._bottleneck_layers = nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, out_channels=out_channels * bottleneck_width, kernel_size=1, bias=True, stride=strides),
                    nn.BatchNorm1d(out_channels * bottleneck_width),
                    Activation(**activation))

            if self.bottleneck == "after":
                if order == "bn_act_conv":
                    self._bottleneck_layers.extend([
                        nn.BatchNorm1d(bottleneck_in_channels),
                        Activation(**activation),
                        nn.Conv1d(in_channels=bottleneck_in_channels, out_channels=out_channels, kernel_size=kernel, bias=True, stride=strides)
                    ])
                elif order == "conv_bn_act":
                    self._bottleneck_layers.extend([
                        nn.Conv1d(in_channels=bottleneck_in_channels, out_channels=out_channels, kernel_size=kernel, bias=True, stride=strides),
                        nn.BatchNorm1d(out_channels),
                        Activation(**activation)
                    ])
            if dropout_rate:
                self._bottleneck_layers.append(nn.Dropout1d(p=dropout_rate))

        if order == "bn_act_conv":
            self._layers = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                Activation(**activation),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, bias=True, stride=strides, dilation=dilation),
            )
        elif order == "conv_bn_act":
            self._layers = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, bias=True, stride=strides, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                Activation(**activation)
            )
        elif order == "conv_act":
            self._layers = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, bias=True, stride=strides, dilation=dilation),
                Activation(**activation)
            )
        elif order == "bn_act":
            self._layers = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                Activation(**activation)
            )
        else:
            raise AssertionError(f"Unknown order {order}")
        if dropout_rate:
            self._layers.append(nn.Dropout1d(p=dropout_rate))

    def forward(self, x):
        y = x
        if self.bottleneck == "before":
            y = self._bottleneck_layers(x)
        y = self._layers(y)
        if self.bottleneck == "after":
            y = self._bottleneck_layers(x)
        return y