from functools import reduce  # Pour appliquer une fonction cumulativement sur un itérable
from operator import mul      # Pour effectuer des multiplications cumulatives
import torch  # Bibliothèque pour le calcul numérique et l'apprentissage profond
import torch.nn as nn  # Modules pour construire des réseaux de neurones

class BaseModule(nn.Module):
    """
    Implémente le module de base.
    Tous les autres modules héritent de celui-ci.
    """
    def load_w(self, checkpoint_path):
        """
        Charge un point de contrôle dans le state_dict.
        :param checkpoint_path: le fichier du point de contrôle à charger.
        """
        self.load_state_dict(torch.load(checkpoint_path))

    def __repr__(self):
        good_old = super(BaseModule, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)
        return good_old

    def __call__(self, *args, **kwargs):
        return super(BaseModule, self).__call__(*args, **kwargs)

    @property
    def n_parameters(self):
        """
        Nombre de paramètres du modèle.
        """
        n_parameters = 0
        for p in self.parameters():
            if hasattr(p, 'mask'):
                n_parameters += torch.sum(p.mask).item()
            else:
                n_parameters += reduce(mul, p.shape)
        return int(n_parameters)

class MaskedConv3d(BaseModule, nn.Conv3d):
    """
    Implémente une convolution 3D masquée.
    Cette convolution 3D ne peut pas accéder aux frames futures.
    """
    def __init__(self, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kT, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kT // 2 + 1:] = 0

    def forward(self, x):
        """
        Effectue le passage avant de la convolution.
        :param x: le tenseur d'entrée.
        :return: le tenseur de sortie après convolution.
        """
        self.weight.data *= self.mask
        return super(MaskedConv3d, self).forward(x)

class TemporallySharedFullyConnection(BaseModule):
    """
    Implémente une connexion entièrement partagée temporellement.
    Traite une série temporelle de vecteurs de caractéristiques et effectue
    la même projection linéaire sur tous.
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        Constructeur de la classe.
        :param in_features: nombre de caractéristiques d'entrée.
        :param out_features: nombre de caractéristiques de sortie.
        :param bias: si True, ajoute un biais.
        """
        super(TemporallySharedFullyConnection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x):
        """
        Fonction forward.
        :param x: entrée de la couche, de forme (batch_size, seq_len, in_features).
        :return: sortie de la couche, de forme (batch_size, seq_len, out_features).
        """
        b, t, d = x.size()
        output = []
        for i in range(t):
            output.append(self.linear(x[:, i, :]))
        output = torch.stack(output, 1)
        return output

def residual_op(x, functions, bns, activation_fn):
    """
    Implémente une opération résiduelle globale.
    :param x: le tenseur d'entrée.
    :param functions: une liste de fonctions (nn.Modules).
    :param bns: une liste de couches batch-norm optionnelles.
    :param activation_fn: l'activation à appliquer.
    :return: la sortie de l'opération résiduelle.
    """
    f1, f2, f3 = functions
    bn1, bn2, bn3 = bns

    assert len(functions) == len(bns) == 3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)

    # Branche A
    ha = x
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)
    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)
    
    # Branche B
    hb = x
    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)
    
    out = ha + hb
    return activation_fn(out)

class BaseBlock(BaseModule):
    """Base class for all blocks."""
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=True):
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(BaseBlock, self).__init__()
        assert not (use_bn and use_bias), 'Using bias=True with batch_normalization is forbidden.'
        self._channel_in = channel_in
        self._channel_out = channel_out
        self._activation_fn = activation_fn
        self._use_bn = use_bn
        self._bias = use_bias

    def get_bn(self):
        """
        Returns batch norm layers if needed, otherwise None.
        """
        return nn.BatchNorm3d(num_features=self._channel_out) if self._use_bn else None

    def forward(self, x):
        """
        Abstract forward function. Not implemented here.
        """
        raise NotImplementedError

class DownsampleBlock(BaseBlock):
    """Implements a Downsampling block for videos."""
    def __init__(self, channel_in, channel_out, activation_fn, stride, use_bn=True, use_bias=False):
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param stride: stride to downsample feature maps.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(DownsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
        self.stride = stride
        self.conv1a = MaskedConv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, padding=1, stride=stride, bias=use_bias)
        self.conv1b = MaskedConv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, padding=0, stride=stride, bias=use_bias)
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x):
        """
        Forward propagation.
        :param x: input tensor.
        :return: output tensor.
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )

class UpsampleBlock(BaseBlock):
    """Implements an Upsampling block for videos."""
    def __init__(self, channel_in, channel_out, activation_fn, stride, output_padding, use_bn=True, use_bias=False):
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param stride: stride to upsample feature maps.
        :param output_padding: padding to be added to the output feature maps.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(UpsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)
        self.stride = stride
        self.output_padding = output_padding
        self.conv1a = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=5, padding=2, stride=stride, output_padding=output_padding, bias=use_bias)
        self.conv1b = nn.Conv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=5, padding=2, stride=stride, output_padding=output_padding, bias=use_bias)
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x):
        """
        Forward propagation.
        :param x: input tensor.
        :return: output tensor.
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )
