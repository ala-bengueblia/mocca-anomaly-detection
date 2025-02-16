from typing import List, Optional
from operator import mul
from functools import reduce
import torch
import torch.nn as nn
from torch.nn import Module


class BaseModule(nn.Module):
    """
    Implements the basic module.
    All other modules inherit from this one.
    """
    
    def load_w(self, checkpoint_path):
        # type: (str) -> None
        """
        Loads a checkpoint into the state_dict.
        :param checkpoint_path: the checkpoint file to be loaded.
        """
        self.load_state_dict(torch.load(checkpoint_path))

    def __repr__(self):
        # type: () -> str
        """
        String representation of the module, including the total number of parameters.
        """
        good_old = super(BaseModule, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)
        return good_old + '\n' + addition

    def __call__(self, *args, **kwargs):
        """
        Calls the base module's __call__ function.
        """
        return super(BaseModule, self).__call__(*args, **kwargs)

    @property
    def n_parameters(self):
        # type: () -> int
        """
        Returns the number of parameters of the model.
        It counts the total number of parameters, considering optional masks.
        """
        n_parameters = 0
        for p in self.parameters():
            if hasattr(p, 'mask'):
                n_parameters += torch.sum(p.mask).item()  # Count only masked parameters
            else:
                n_parameters += reduce(mul, p.shape)  # Multiply dimensions of each parameter tensor
        return int(n_parameters)


def residual_op(x, functions, bns, activation_fn):
    # type: (torch.Tensor, List[Module, Module, Module], List[Module, Module, Module], Module) -> torch.Tensor
    """
    Implements a global residual operation.
    :param x: the input tensor.
    :param functions: a list of functions (nn.Modules).
    :param bns: a list of optional batch-norm layers.
    :param activation_fn: the activation to be applied.
    :return: the output of the residual operation.
    """
    f1, f2, f3 = functions
    bn1, bn2, bn3 = bns

    # Check that we have exactly 3 functions and batch-norm layers
    assert len(functions) == len(bns) == 3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)

    # A-branch: Sequence of convolutions and batch-normalization
    ha = x
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)

    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)

    # B-branch: Skip connection with possible convolution and batch-normalization
    hb = x
    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)

    # Residual connection: Adding the A and B branches
    out = ha + hb
    return activation_fn(out)


class BaseBlock(BaseModule):
    """ 
    Base class for all blocks.
    """
    
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation function to be used.
        :param use_bn: whether or not to use batch normalization.
        :param use_bias: whether or not to use bias.
        """
        super(BaseBlock, self).__init__()

        assert not (use_bn and use_bias), 'Using bias=True with batch_normalization is forbidden.'

        # Initialize block parameters
        self._channel_in = channel_in
        self._channel_out = channel_out
        self._activation_fn = activation_fn
        self._use_bn = use_bn
        self._bias = use_bias

    def get_bn(self):
        # type: () -> Optional[Module]
        """
        Returns batch normalization layers, if needed.
        :return: batch normalization layers or None.
        """
        return nn.BatchNorm2d(num_features=self._channel_out) if self._use_bn else None

    def forward(self, x):
        """
        Abstract forward function. Must be implemented in subclasses.
        """
        raise NotImplementedError


class DownsampleBlock(BaseBlock):
    """ 
    Implements a downsampling block for images (used for downscaling) in CASIA1.
    """
    
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation function to be used.
        :param use_bn: whether or not to use batch normalization.
        :param use_bias: whether or not to use bias.
        """
        super(DownsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Define convolution layers
        self.conv1a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, padding=0, stride=2, bias=use_bias)

        # Define batch normalization layers
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation for downsampling.
        :param x: input tensor
        :return: output tensor after downsampling
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )


class UpsampleBlock(BaseBlock):
    """ 
    Implements an upsampling block for images (used for upscaling) in CASIA1.
    """
    
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation function to be used.
        :param use_bn: whether or not to use batch normalization.
        :param use_bias: whether or not to use bias.
        """
        super(UpsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Define convolution layers
        self.conv1a = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1, bias=use_bias)
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=1, padding=0, stride=2, output_padding=1, bias=use_bias)

        # Define batch normalization layers
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation for upsampling.
        :param x: input tensor
        :return: output tensor after upsampling
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )


class ResidualBlock(BaseBlock):
    """ 
    Implements a residual block for images (used for feature extraction) in CASIA1.
    """
    
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation function to be used.
        :param use_bn: whether or not to use batch normalization.
        :param use_bias: whether or not to use bias.
        """
        super(ResidualBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Define convolution layers
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv2 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1, bias=use_bias)

        # Define batch normalization layers
        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation for residual block.
        :param x: input tensor
        :return: output tensor after residual operation
        """
        return residual_op(
            x,
            functions=[self.conv1, self.conv2, None],
            bns=[self.bn1, self.bn2, None],
            activation_fn=self._activation_fn
        )
