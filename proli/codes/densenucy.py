"""Dense Pafnucy class definition : Densenucy.

Code porting of `https://gitlab.com/cheminfIBB/pafnucy` to pytorch/1.11
DenseNet version of Pafnucy
DenseNet : https://pytorch.org/hub/pytorch_vision_densenet

    Typical usage examples:

    model = get_densenucy()
    features = make_features(growth_rate, dense_cfg)
    regressor = make_regressor(fc_cfg)
    model2 = Densenucy(features, regressor)
"""

# --- built-in python module
from typing import List

# --- external modules
import torch
import torch.nn as nn


class Densenucy(nn.Module):
    """3D Dense Convolutionnal Neural Network for Affinity prediction of
    Protein-Ligand complexes.

    Attributes:
        features (nn.Module): part of Densenucy in charge of extracting
                              deep spatial features
        regressor (nn.Module): part of Densenucy in charge of predicting
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


def make_conv_block(input_channels, output_channels):
    """Create a conv block. 
    
    The conv block applies the following operations : 
        y = relu(batch_norm(conv3d(x, w) + b))
    where x = input, w = weights, b = bias and y = output.
    
    Args: 
        input_channels : number of input channels
        output_channels : number of output channels
    """
    return nn.Sequential(
        nn.Conv3d(input_channels, output_channels, kernel_size=3, padding='same'),
        nn.BatchNorm3d(output_channels), nn.ReLU())


class DenseBlock(nn.Module):
    """Define the dense block used in DenseNet.

    Attributes:
        conv_blocks (nn.Module): a sequence of conv blocks used in the class
    """
    
    def __init__(self, n_blocks, input_channels, growth_rate):
        super().__init__()
        self.conv_blocks = nn.Sequential()
        for i in range(n_blocks):
            self.conv_blocks.add_module(
                f'conv_block_{i+1}',
                make_conv_block(growth_rate*i + input_channels, growth_rate))

    def forward(self, x):
        for block in self.conv_blocks:
            y = block(x)
            # Concatenate the input and output of each block on the 
            # channel dimension
            x = torch.cat((x, y), dim=1)
        return x

    
def make_transition_block(input_channels, output_channels):
    """Create a transition block. 
    
    Reduce the number of channels and the image size. Compress the 
    spatials information and the channels information.
    
    Args: 
        input_channels : number of input channels
        output_channels : number of output channels
    """
    return nn.Sequential(
        nn.Conv3d(input_channels, output_channels, kernel_size=1),
        nn.BatchNorm3d(output_channels), nn.ReLU(),
        nn.AvgPool3d(kernel_size=2, stride=2, ceil_mode=True))


def make_features(growth_rate, n_conv_blocks_in_dense_blocks):
    """Create a 3D spatial features extractor for Densenucy.
    
    The extractor is made of an input block and a sequence of dense block
    and transition block.
    A dense block is made of several conv block.
    After passing each conv block in a dense block, the new information
    are added to the input of the dense block, which helps in conserving 
    information through the passage of the dense block. Useful when 
    dealing with sparse input tensor such as the Protein-Ligand affinity
    prediction use-case.
    
    Arg examples :
        growth_rate = 96
        n_conv_blocks_in_dense_blocks = [4 ,3]
    In this example, we have 1 dense block with 4 conv blocks and 
    1 dense block with 3 conv blocks. 
    Since the first dense block has 4 conv blocks, 
    n_channels output first dense block = growth_rate*4 + n_channels intput
    
    Args: 
        growth_rate : n_channels added after each conv block in dense block
        n_conv_blocks_in_dense_blocks : number of conv blocks for dense block
    """
    input_block = nn.Sequential(
        nn.Conv3d(19, 64, kernel_size=3, padding='same'),
        nn.BatchNorm3d(64), nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, ceil_mode=True))
    
    features = nn.Sequential()
    features.add_module('input_block', input_block)
    # `n_channels`: the current number of channels
    n_channels = 64
    for i, n_conv_blocks in enumerate(n_conv_blocks_in_dense_blocks, 1):
        features.add_module(
            f'dense_block_{i}',
            DenseBlock(n_conv_blocks, n_channels, growth_rate))
        # This is the number of output channels in the previous dense block
        n_channels += n_conv_blocks * growth_rate
        # A transition layer, that is halving the number of channels 
        # and the image size, is added
        features.add_module(
            f'transition_block_{i}',
            make_transition_block(n_channels, n_channels // 2))
        n_channels = n_channels // 2
    
    return features


def make_regressor(fc_cfg: List[int]) -> nn.Sequential:
    """Create a multilayer perceptron for a regression task.
    
    The MLP applies several fully connected blocks and an output layer.
    Each fc block applies the following transformation :
        y = relu(batch_norm(matmut(x, w) + b))
    where x = input, w = weights, b = bias and y = output.
    The output layer is a `nn.Linear` with a single neuron output.
    
    Arg examples :
        fc_cfg = [input_neurons, ..., neurons_before_output_layer]
    Create a sequence of `len(fc_cfg)-1` fc blocks and the output layer.
    
    Args: 
        fc_cfg : number of neurons before and after each fc block
    """
    regressor = nn.Sequential()
    in_neurons = fc_cfg[0]
    for i, out_neurons in enumerate(fc_cfg[1:], 1):
        fc_block = nn.Sequential()
        linear = nn.Linear(in_neurons, out_neurons)
        batchnorm = nn.BatchNorm1d(num_features=out_neurons)
        relu = nn.ReLU()
        fc_block.add_module('linear', linear)
        fc_block.add_module('batchnorm', batchnorm)
        fc_block.add_module('relu', relu)
        regressor.add_module(f'fc_block_{i}', fc_block)
        in_neurons = out_neurons
    regressor.add_module(f'output_layer', nn.Linear(fc_cfg[-1], 1))
    
    return regressor


# default input tensor is of shape (19,25,25,25)
# each input is a grid of 25x25x25
# with 19 features for each cell in the grid
default_dense_cfg = [2, 2]
default_fc_cfg = [160 * 4 * 4 * 4]


def create_densenucy(    
    growth_rate: int=96,
    dense_cfg: List[int]=default_dense_cfg,
    fc_cfg: List[int]=default_fc_cfg, 
    pretrained_path: str=None
) -> Densenucy:
    """Create a densenucy model with/without pretrained weights.
    
    Args: 
        growth_rate : number of added channels after each conv block 
                      inside of the dense blocks 
                      cf. DenseNet architecture
        dense_cfg : number of conv blocks for each dense block
        fc_cfg : number of neurons before and after each fc block
        pretrained_path : path to charge pretrained weights
    """
    features = make_features(growth_rate, dense_cfg)
    regressor = make_regressor(fc_cfg)

    if pretrained_path and isinstance(pretrained_path, str):
        model = Densenucy(features, regressor, init_weights=False)
        model.load_state_dict(torch.load(pretrained_path))
    else:
        model = Densenucy(features, regressor)
    return model