import sys
from operator import mul
from typing import Tuple
from functools import reduce
# Importation des bibliothèques nécessaires
import torch
import torch.nn as nn
import torch.nn.functional as F
# Importation des modules personnalisés nécessaires
from .micc_f220_base_model import BaseModule, DownsampleBlock, ResidualBlock, UpsampleBlock

# Définition des canaux pour les différentes couches
CHANNELS = [32, 64, 128]

def init_conv_blocks(channel_in: int, channel_out: int, activation_fn: nn.Module) -> nn.Module:
    """Init des blocs convolutionnels.

    Parameters
    ----------
    channel_in : int
        Nombre de canaux en entrée.
    channel_out : int
        Nombre de canaux de sortie.
    activation_fn : nn.Module
        Fonction d'activation à utiliser.

    Returns
    -------
    nn.Module
        Bloc de downsampling initialisé.
    """
    return DownsampleBlock(channel_in=channel_in, channel_out=channel_out, activation_fn=activation_fn)

class Selector(nn.Module):
    """Module Selector."""
    def __init__(self, code_length: int, idx: int):
        super().__init__()
        """Initialisation de l'architecture Selector

        Parameters
        ----------
        code_length : int
            Taille du code latent.
        idx : int
            Indice de la couche.
        """
        # Liste des profondeurs des cartes de caractéristiques
        sizes = [CHANNELS[0], CHANNELS[0], CHANNELS[1], CHANNELS[2], CHANNELS[2]*2, CHANNELS[2]*2, code_length]
        
        # Taille de la sortie du FC caché
        mid_features_size = 256

        # Taille de la sortie du dernier FC
        out_features = 128

        # Choisir une architecture Selector différente selon la couche à laquelle il est attaché
        if idx < 5:
            self.fc = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=8),
                nn.Conv2d(in_channels=sizes[idx], out_channels=1, kernel_size=1),
                nn.Flatten(),
                nn.Linear(in_features=8**2, out_features=mid_features_size, bias=True),
                nn.BatchNorm1d(mid_features_size),
                nn.ReLU(),
                nn.Linear(in_features=mid_features_size, out_features=out_features, bias=True)
            )
        else:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=sizes[idx],  out_features=mid_features_size, bias=True),
                nn.BatchNorm1d(mid_features_size),
                nn.ReLU(),
                nn.Linear(in_features=mid_features_size, out_features=out_features, bias=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class MICC_F220_Encoder(BaseModule):
    """Encodeur du réseau MICC_F220."""
    def __init__(self, input_shape: torch.Tensor, code_length: int, idx_list_enc: list, use_selectors: bool):
        """Initialisation du réseau Encodeur
        
        Parameters
        ----------
        input_shape : torch.Tensor
            Forme des données d'entrée.
        code_length : int
            Taille du code latent.
        idx_list_enc : list 
            Liste des indices de couches à utiliser pour la tâche AD.
        use_selectors : bool
            Vrai si le modèle doit utiliser les modules Selectors, Faux sinon.
        """
        super().__init__()
        
        self.idx_list_enc = idx_list_enc
        self.use_selectors = use_selectors

        # Forme des données d'entrée
        c, h, w = input_shape

        # Fonction d'activation
        self.activation_fn = nn.LeakyReLU()

        # Initialisation des blocs convolutionnels
        self.conv = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, bias=False)
        self.res  = ResidualBlock(channel_in=32, channel_out=32, activation_fn=self.activation_fn)
        self.dwn1, self.dwn2, self.dwn3 = [init_conv_blocks(channel_in=ch, channel_out=ch*2, activation_fn=self.activation_fn) for ch in CHANNELS]
        
        # Profondeur de la dernière carte de caractéristiques
        self.last_depth = CHANNELS[2]*2

        # Forme de la dernière carte de caractéristiques
        self.deepest_shape = (self.last_depth, h // 8, w // 8)
        
        # Initialisation des couches FC
        self.fc1 = nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=self.last_depth)
        self.bn  = nn.BatchNorm1d(num_features=self.last_depth)
        self.fc2 = nn.Linear(in_features=self.last_depth, out_features=code_length)
        
        ## Initialisation des modèles Selector
        if self.use_selectors:
            self.selectors = nn.ModuleList([Selector(code_length=code_length, idx=idx) for idx in range(7)])
            self.selectors.append(Selector(code_length=code_length, idx=6))

    def get_depths_info(self) -> [int, Tuple[int, int, int]]:
        """
        Renvoie
        ------
        self.last_depth : int
            Profondeur de la dernière carte de caractéristiques.
        self.deepest_shape : tuple
            Forme de la dernière carte de caractéristiques.
        """
        return self.last_depth, self.deepest_shape

    def set_idx_list_enc(self, idx_list_enc: list) -> None:
        """Définit la liste des couches à partir desquelles extraire les caractéristiques.

        Parameters
        ----------
        idx_list_enc : list
            Liste des indices des couches.
        """
        self.idx_list_enc = idx_list_enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1 = self.conv(x)
        o2 = self.res(self.activation_fn(o1))
        o3 = self.dwn1(o2)
        o4 = self.dwn2(o3)
        o5 = self.dwn3(o4)
        o7 = self.activation_fn(self.bn(self.fc1(o5.view(len(o5), -1))))  # FC -> BN -> LeakyReLU
        o8 = self.fc2(o7)
        z = nn.Sigmoid()(o8)
        
        outputs = [o1, o2, o3, o4, o5, o7, o8, z]

        if len(self.idx_list_enc) != 0:
            if self.use_selectors:
                tuple_o = [self.selectors[idx](tt) for idx, tt in enumerate(outputs) if idx in self.idx_list_enc]
            else:
                tuple_o = []
                for idx, tt in enumerate(outputs):
                    if idx not in self.idx_list_enc:
                        continue
                    if tt.ndimension() > 2:
                        tuple_o.append(F.avg_pool2d(tt, tt.shape[-2:]).squeeze())
                    else:
                        tuple_o.append(tt.squeeze())
            return list(zip([f'0{idx}' for idx in self.idx_list_enc], tuple_o))
        else:
            return z

class MICC_F220_Decoder(BaseModule):
    """Décodeur du réseau MICC_F220."""
    def __init__(self, code_length: int, deepest_shape: Tuple[int, int, int], last_depth: int, output_shape: torch.Tensor):
        """Initialisation du réseau Décodeur
        
        Parameters
        ----------
        code_length : int
            Taille du code latent.
        deepest_shape : tuple
            Forme de la dernière carte de caractéristiques de l'encodeur.
        last_depth : int
            Profondeur de la dernière carte de caractéristiques de l'encodeur.
        output_shape : torch.Tensor
            Forme des données de sortie.
        """
        super().__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        # Fonction d'activation du décodeur
        activation_fn = nn.LeakyReLU()

        # Réseau FC
        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=last_depth),
            nn.BatchNorm1d(num_features=last_depth),
            activation_fn,
            nn.Linear(in_features=last_depth, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        # Réseau (Transposé) Convolutionnel
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=CHANNELS[2]*2, channel_out=CHANNELS[2], activation_fn=activation_fn),
            UpsampleBlock(channel_in=CHANNELS[1]*2, channel_out=CHANNELS[1], activation_fn=activation_fn),
            UpsampleBlock(channel_in=CHANNELS[0]*2, channel_out=CHANNELS[0], activation_fn=activation_fn),
            ResidualBlock(channel_in=CHANNELS[0], channel_out=CHANNELS[0], activation_fn=activation_fn),
            nn.Conv2d(in_channels=CHANNELS[0], out_channels=3, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        h = h.view(len(h), *self.deepest_shape)
        return self.conv(h)

class MICC_F220_AutoEncoder(BaseModule):
    """Réseau AutoEncoder complet MICC_F220."""
    def __init__(self, input_shape: torch.Tensor, code_length: int, use_selectors: bool):
        """Initialisation du Full AutoEncoder

        Parameters
        ----------
        input_shape : torch.Tensor
            Forme des données d'entrée.
        code_length : int
            Taille du code latent.
        use_selectors : bool
            Vrai si le modèle doit utiliser les modules Selectors, Faux sinon.
        """
        super().__init__()

        # Forme des données d'entrée requise par le décodeur
        self.input_shape = input_shape
        
        # Construction de l'encodeur
        self.encoder = MICC_F220_Encoder(
            input_shape=input_shape,
            code_length=code_length,
            idx_list_enc=[],
            use_selectors=use_selectors
        )

        last_depth, deepest_shape = self.encoder.get_depths_info()

        # Construction du décodeur
        self.decoder = MICC_F220_Decoder(
            code_length=code_length,
            deepest_shape=deepest_shape,
            last_depth=last_depth,
            output_shape=input_shape
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)
        return x_r
