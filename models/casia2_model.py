import logging  # Pour la gestion des messages d'information ou d'erreur
import numpy as np  # Pour les opérations mathématiques et la manipulation de tableaux
import torch  # Pour la manipulation des tenseurs et les calculs sur GPU
import torch.nn as nn  # Pour construire les réseaux de neurones avec PyTorch
import torch.nn.functional as F  # Pour des fonctions complémentaires des réseaux de neurones

# Fonction d'initialisation d'une couche convolutionnelle
def init_conv(out_channels: int, k_size: int = 5) -> nn.Module:
    """Initialise une couche convolutionnelle.

    Parameters
    ----------
    k_size : int
        Taille du noyau (kernel size).
    out_channels : int
        Nombre de canaux de sortie (dimension des features).

    Returns
    -------
    nn.Module
        Couche Conv2d initialisée.
    """
    conv = nn.Conv2d(
        in_channels=3 if out_channels == 32 else out_channels // 2,  # 3 canaux en entrée pour la première couche, sinon out_channels//2
        out_channels=out_channels,
        kernel_size=k_size,
        bias=False,
        padding=2  # Pour conserver la même taille en sortie
    )
    nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain('leaky_relu'))
    return conv

# Fonction d'initialisation d'une couche deconvolutionnelle
def init_deconv(out_channels: int, k_size: int = 5) -> nn.Module:
    """Initialise une couche deconvolutionnelle (transposée).

    Parameters
    ----------
    k_size : int
        Taille du noyau.
    out_channels : int
        Nombre de canaux de sortie.

    Returns
    -------
    nn.Module
        Couche ConvTranspose2d initialisée.
    """
    deconv = nn.ConvTranspose2d(
        in_channels=out_channels,
        out_channels=3 if out_channels == 32 else out_channels // 2,
        kernel_size=k_size,
        bias=False,
        padding=2
    )
    nn.init.xavier_uniform_(deconv.weight, gain=nn.init.calculate_gain('leaky_relu'))
    return deconv

# Fonction d'initialisation d'une couche de normalisation par lot (BatchNorm)
def init_bn(num_features: int) -> nn.Module:
    """Initialise une couche BatchNorm2d.

    Parameters
    ----------
    num_features : int
        Nombre de caractéristiques en entrée.

    Returns
    -------
    nn.Module
        Couche BatchNorm2d initialisée.
    """
    return nn.BatchNorm2d(num_features=num_features, eps=1e-04, affine=False)

# Classe de base pour les réseaux de neurones
class BaseNet(nn.Module):
    """Classe de base pour tous les réseaux de neurones."""
    def __init__(self):
        super(BaseNet, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        # Liste des tailles de sorties pour les couches convolutionnelles
        self.output_features_sizes = [32, 64, 128]

    def summary(self) -> None:
        """Affiche un résumé du modèle."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

# Définition de l'encodeur pour CASIA2
class CASIA2_Encoder(BaseNet):
    """Encodeur pour le réseau CASIA2."""
    def __init__(self, code_length: int):
        """Initialise l'encodeur.

        Parameters
        ----------
        code_length : int
            Taille du code latent.
        """
        super(CASIA2_Encoder, self).__init__()
        # Initialisation des couches convolutionnelles
        self.conv1, self.conv2, self.conv3 = [init_conv(out_channels) for out_channels in self.output_features_sizes]
        # Initialisation des couches BatchNorm correspondantes
        self.bnd1, self.bnd2, self.bnd3 = [init_bn(num_features) for num_features in self.output_features_sizes]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Couche entièrement connectée pour obtenir le code latent
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=code_length, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bnd1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bnd2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bnd3(x)))
        x = self.fc1(x.view(x.size(0), -1))
        return x

# Définition du décodeur pour CASIA2
class CASIA2_Decoder(BaseNet):
    """Décodeur complet pour CASIA2."""
    def __init__(self, code_length: int):
        """Initialise le décodeur.

        Parameters
        ----------
        code_length : int
            Taille du code latent.
        """
        super(CASIA2_Decoder, self).__init__()
        self.rep_dim = code_length
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        # Construction du décodeur avec une première couche deconvolutionnelle
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        # Initialisation des autres couches deconvolutionnelles dans l'ordre décroissant des features
        self.deconv2, self.deconv3, self.deconv4 = [init_deconv(out_channels) for out_channels in self.output_features_sizes[::-1]]
        # Couches BatchNorm associées aux deconv
        self.bnd4, self.bnd5, self.bnd6 = [init_bn(num_features) for num_features in self.output_features_sizes[::-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1d(x)
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bnd4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bnd5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bnd6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x

# Définition de l'autoencodeur complet pour CASIA2
class CASIA2_Autoencoder(BaseNet):
    """Autoencodeur complet pour CASIA2."""
    def __init__(self, code_length: int = 128):
        """Initialise l'autoencodeur.

        Parameters
        ----------
        code_length : int, optional
            Taille du code latent (par défaut 128).
        """
        super(CASIA2_Autoencoder, self).__init__()
        self.encoder = CASIA2_Encoder(code_length=code_length)
        self.bn1d = nn.BatchNorm1d(num_features=code_length, eps=1e-04, affine=False)
        self.decoder = CASIA2_Decoder(code_length=code_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.bn1d(z)
        return self.decoder(z)
