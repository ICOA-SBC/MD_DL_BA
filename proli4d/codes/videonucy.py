"""Video Pafnucy class definition : Videonucy.

Code porting of `https://gitlab.com/cheminfIBB/pafnucy` to pytorch/1.11
ConvLSTM version of Pafnucy, processing a video instead of an image
ConvLSTM : https://github.com/ndrplz/ConvLSTM_pytorch

    Typical usage examples:

    >> model = get_videonucy()
    >> features = make_features(convlstm_cfg, 5, False)
    >> regressor = make_regressor(fc_cfg)
    >> model2 = Videonucy(features, regressor)
"""

# --- built-in python module
from typing import List, Tuple
from math import ceil

# --- external modules
import torch
import torch.nn as nn


class Videonucy(nn.Module):
    """ConvLSTM Neural Network for Affinity prediction of Protein-Ligand
    complexes.

    Attributes:
        features (nn.Module): part of Videonucy in charge of extracting
                              deep spatiotemporal features
        regressor (nn.Module): part of Videonucy in charge of predicting
                               affinity from the extracted features
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
                if isinstance(l, nn.Linear):
                    nn.init.kaiming_normal_(l.weight,
                                            mode="fan_in",
                                            nonlinearity="relu")
                    nn.init.constant_(l.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.features(x)
        # -1 for the last ConvLSTMCell and -1 for the last frame
        x = outputs[-1][:, -1, :, :, :, :]
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        # we expect that the output tensor is of shape (N,1)
        # let's remove the unnecessary dimension : (N,1) -> (N,)
        return x.squeeze()
    

class ConvLSTMCell(nn.Module):

    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        conv_kernel_size: int
    ) -> None:
        """Initialize ConvLSTM cell.
        
        Args:
            in_channels : number of input channels
            out_channels : number of output channels
            conv_kernel_size : size of the convolutional kernel
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Conv3d(in_channels  = self.in_channels + self.out_channels,
                              out_channels = 4 * self.out_channels,
                              kernel_size  = conv_kernel_size,
                              padding      = 'same')

    def forward(
        self, 
        input_tensor: torch.Tensor, 
        cur_state: Tuple[torch.Tensor,torch.Tensor]
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.out_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden_state(
        self, 
        batch_size: int, 
        image_size: Tuple[int,int,int]
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        """Init the long and short term memories
        """
        depth, height, width = image_size
        return (torch.zeros(batch_size, self.out_channels, depth, height, width, 
                            device=self.conv.weight.device),
                torch.zeros(batch_size, self.out_channels, depth, height, width, 
                            device=self.conv.weight.device))


class ConvLSTMBlock(nn.Module):

    """Create a sequence of (ConvLSTMCell + max_pooling) layers. 
    
    Each layer applies the following transformation :
        y = max_pool(ConvLSTMCell(x))
    where x = input and y = output.
    
    Args:
        convlstm_cfg: number of channels before and after each ConvLSTMCell
        conv_kernel_size: size of the convolution kernel in each ConvLSTMCell
        return_all_layers: return the list of computations for all layers
    Input:
        tensor of size (B, T, C, D, H, W)
        B=batch size
        T=number of frames in the video
        C=channel  D=depth  H=height  W=wide
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 19, 25, 25, 25))
        >> convlstm = ConvLSTMBlock([19, 64, 128, 256], 5, False)
        >> layer_output, _ = convlstm(x)
        # -1 for the output of the last ConvLSTMCell, and take the last output frame
        >> y = layer_output[-1][:, -1, :, :, :, :] 
    """

    def __init__(
        self, 
        convlstm_cfg: List[int],
        conv_kernel_size: int,
        return_all_layers: bool=False
    ) -> None:
        super().__init__()

        self.return_all_layers = return_all_layers
        self.num_layers = len(convlstm_cfg) - 1
        
        cell_list = []
        in_channels = convlstm_cfg[0]
        for out_channels in convlstm_cfg[1:]:
            cell_list.append(ConvLSTMCell(in_channels=in_channels,
                                          out_channels=out_channels,
                                          conv_kernel_size=conv_kernel_size))
            in_channels = out_channels
        
        self.cell_list = nn.ModuleList(cell_list)
        self.maxpool = nn.MaxPool3d(2, ceil_mode=True)
        

    def forward(self, input_tensor: torch.Tensor) -> Tuple[List,List]:
        """
        Args:
            input_tensor : 6-D Tensor of shape (b, t, c, d, h, w)
    
        Returns:
            layer_output, last_state_list
        """

        b, _, _, d, h, w = input_tensor.size()

        # Init hidden states of all ConvLSTMCell
        hidden_state = self._init_hidden(batch_size=b,
                                         image_size=(d, h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx, cell in enumerate(self.cell_list):

            h, c = hidden_state[layer_idx]
            output_maxpool = []
            for t in range(seq_len):
                h, c = cell(input_tensor=cur_layer_input[:, t, :, :, :, :],
                            cur_state=[h, c])
                output_maxpool.append(self.maxpool(h))

            # from a list of 5D tensor, create a 6D tensor
            maxpool_output = torch.stack(output_maxpool, dim=1)
            # input for the next layer
            cur_layer_input = maxpool_output

            layer_output_list.append(maxpool_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(
        self, 
        batch_size: int, 
        image_size: Tuple[int,int,int]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Init the hidden states for all ConvLSTMCell
        """
        input_size = image_size
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden_state(batch_size, 
                                                                   input_size))
            # each ConvLSTMCell is followed by a pooling operation
            # reducing the size of the input for the next ConvLSTMCell
            d, h, w = input_size
            input_size = (ceil(d/2), ceil(h/2), ceil(w/2))
            
        return init_states
    

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


# default input tensor is of shape (50,19,25,25,25)
# each input is a grid of 25x25x25
# with 19 features for each cell in the grid
# and 50 frames in the video
default_convlstm_cfg = [19, 64, 128, 256]
default_fc_cfg = [256 * 4 * 4 * 4, 1024, 256]


def create_videonucy(    
    convlstm_cfg: List[int]=default_convlstm_cfg,
    fc_cfg: List[int]=default_fc_cfg, 
    pretrained_path: str=None
) -> Videonucy:
    """Create a videonucy model with/without pretrained weights.
    
    Args:
        convlstm_cfg: number of channels before and after each ConvLSTMCell
        fc_cfg : number of neurons before and after each fc block
        pretrained_path : path to charge pretrained weights
    """
    features = ConvLSTMBlock(convlstm_cfg, 5, False)
    regressor = make_regressor(fc_cfg)

    if pretrained_path and isinstance(pretrained_path, str):
        model = Videonucy(features, regressor, init_weights=False)
        model.load_state_dict(torch.load(pretrained_path))
    else:
        model = Videonucy(features, regressor)
    return model