"""Pafnucy class definition.

Code porting of `https://gitlab.com/cheminfIBB/pafnucy` to pytorch/1.11
The initial project used `tfbio.net` module to define Pafnucy in 
tensorflow/1.2 

    Typical usage example:

    model = get_pafnucy()
    features = make_conv_block(conv_cfg, kconv, kpool)
    regressor = make_regressor(fc_cfg, dropout_prob)
    model2 = Pafnucy(features, regressor) 
"""

# --- built-in python module
from typing import List

# --- external modules
import torch
import torch.nn as nn


class Pafnucy(nn.Module):
    """Simple 3DConvolutionnal Neural Network used in Pafnucy paper.

    Attributes:
        features (nn.Module): part of Pafnucy in charge of extracting 
                              deep spatial features
        regressor (nn.Module): part of Pafnucy in charge of predicting 
                               affinity from the deep spatial features
    """

    def __init__(
            self,
            features: nn.Module,
            regressor: nn.Module,
            init_weights: bool = True
    ) -> None:
        super().__init__()
        self.features = features
        self.regressor = regressor
        if init_weights:
            for l in self.modules():
                if isinstance(l, nn.Conv3d):
                    nn.init.kaiming_normal_(l.weight,
                                            mode="fan_in",
                                            nonlinearity="relu")
                    nn.init.constant_(l.bias, 0.1)
                elif isinstance(l, nn.Linear):
                    nn.init.kaiming_normal_(l.weight,
                                            mode="fan_in",
                                            nonlinearity="relu")
                    nn.init.constant_(l.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        # we expect that the output tensor is of shape (N,1)
        # let's remove the unnecessary dimension : (N,1) -> (N,)
        return x.squeeze()


def make_conv_block(
        conv_cfg: List[int],
        conv_kernel_size: int,
        pool_kernel_size: int
) -> nn.Sequential:
    """Create a sequence of (3D_convolution + max_pooling) layer. 
    
    Each layer applies the following transformation :
        y = max_pool(relu(conv3d(x, w) + b))
    where x = input, w = convolution kernel, b = bias and y = output.
    
    Args: 
        conv_cfg : number of filters before and after each convolution
        conv_kernel_size : size of the convolutionnal kernel
        pool_kernel_size : size of the max pooling kernel
    """
    conv_block = nn.Sequential()
    in_channels = conv_cfg[0]
    for i, out_channels in enumerate(conv_cfg[1:], 1):
        conv3d = nn.Conv3d(in_channels,
                           out_channels,
                           kernel_size=conv_kernel_size,
                           padding='same')
        relu = nn.ReLU()
        maxpool = nn.MaxPool3d(pool_kernel_size, ceil_mode=True)
        conv_block.add_module(f'conv{i}_conv3d', conv3d)
        conv_block.add_module(f'conv{i}_relu', relu)
        conv_block.add_module(f'conv{i}_maxpool', maxpool)
        in_channels = out_channels
    return conv_block


def make_fc_block(fc_cfg: List[int], dropout_prob: float) -> nn.Sequential:
    fc_block = nn.Sequential()
    in_neurons = fc_cfg[0]
    for i, out_neurons in enumerate(fc_cfg[1:], 1):
        linear = nn.Linear(in_neurons, out_neurons)
        relu = nn.ReLU()
        dropout = nn.Dropout(p=dropout_prob)
        fc_block.add_module(f'fc{i}_linear', linear)
        fc_block.add_module(f'fc{i}_relu', relu)
        fc_block.add_module(f'fc{i}_dropout', dropout)
        in_neurons = out_neurons
    return fc_block


def make_regressor(fc_cfg: List[int], dropout_prob: float) -> nn.Sequential:
    regressor = make_fc_block(fc_cfg, dropout_prob)
    regressor.add_module(f'output', nn.Linear(fc_cfg[-1], 1))
    return regressor


# default input tensor is of shape (19,25,25,25)
# each input is a grid of 25x25x25
# with 19 features for each cell in the grid
default_input_spatial_dim = 25
default_conv_cfg = [19, 64, 256, 1024]
default_fc_cfg = [1024 * 4 * 4 * 4, 1024 * 4, 1024, 256]


def create_pafnucy(cfg, pretrained_path=None
                   ) -> Pafnucy:
    # setup
    conv_kernel_size = cfg.conv_kernel_size
    pool_kernel_size = cfg.pool_kernel_size
    conv_channels = cfg.conv_channels
    fc_channels = cfg.dense_sizes
    dropout_prob = cfg.kp

    features = make_conv_block(conv_channels, conv_kernel_size, pool_kernel_size)
    regressor = make_regressor(fc_channels, dropout_prob)

    if pretrained_path and isinstance(pretrained_path, str):  # TODO: not implemented
        model = Pafnucy(features, regressor, init_weights=False)
        model.load_state_dict(torch.load(pretrained_path))
    else:
        model = Pafnucy(features, regressor)
    return model


def describe(model):
    print('===========================================\nLayer   \
                                   Param #\n=====================\
                                   ======================')
    for name, l in model.named_modules():
        if isinstance(l, nn.Conv3d):
            print(f'{name}\n                                   \
                {torch.numel(l.weight) + torch.numel(l.bias)}')
        elif isinstance(l, nn.ReLU):
            print(name)
        elif isinstance(l, nn.MaxPool3d):
            print(name)
        elif isinstance(l, nn.Linear):
            print(f'{name}\n                                   \
                {torch.numel(l.weight) + torch.numel(l.bias)}')
        elif isinstance(l, nn.Dropout):
            print(name)
