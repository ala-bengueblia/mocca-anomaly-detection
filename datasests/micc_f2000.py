# Importation des bibliothèques nécessaires
from glob import glob           # Pour rechercher des fichiers via un motif
from tqdm import tqdm           # Pour afficher une barre de progression
from time import time           # Pour mesurer les performances si nécessaire
from typing import List, Tuple  # Pour l'annotation des types
from os.path import basename, isdir, join, splitext  # Pour manipuler les chemins de fichiers
import cv2                      # Bibliothèque OpenCV pour le traitement d'images
import numpy as np              # Pour les opérations sur les tableaux numériques
import skimage.io as io         # Pour la lecture et l'écriture d'images avec skimage
import torch                    # Framework pour l'apprentissage automatique
from torchvision import transforms  # Pour appliquer des transformations aux images
from skimage.transform import resize  # Pour redimensionner les images avec skimage
from torch.utils.data import Dataset, DataLoader  # Pour gérer les ensembles de données PyTorch
from torch.utils.data.dataloader import default_collate  # Pour regrouper les données en batchs
# Importation du gestionnaire de test spécifique au dataset MICC_F2000
from .micc_f2000_test import MICC_F2000TestHandler

# Classe de gestion du dataset MICC_F2000
class MICC_F2000_DataHolder(object):
    """
    Classe de gestion des données pour le dataset MICC_F2000.

    Paramètres
    ----------
    root : str
        Chemin du dossier racine contenant le dataset MICC_F2000.
    clip_length : int
        Nombre d'images formant un clip vidéo.
    stride : int
        Pas pour la création d'un clip (contrôle la superposition des images).
    """
    def __init__(self, root: str, clip_length=16, stride=1):
        self.root = root                              # Racine du dataset
        self.clip_length = clip_length                # Nombre d'images par clip
        self.stride = stride                          # Pas de déplacement entre clips
        self.shape = (3, clip_length, 256, 512)         # Dimensions attendues : (canaux, frames, hauteur, largeur)
        self.train_dir = join(root, 'training', 'nobackground_frames_resized')  # Dossier des images d'entraînement

        # Transformation à appliquer aux vidéos (normalisation et conversion en tenseur)
        self.transform = transforms.Compose([ToFloatTensor3D(normalize=True)])

    def get_test_data(self) -> Dataset:
        """
        Charge le dataset de test et retourne un objet Dataset personnalisé.

        Returns
        -------
        MICC_F2000TestHandler
            Gestionnaire des données de test pour MICC_F2000.
        """
        return MICC_F2000TestHandler(self.root)

    def get_train_data(self, return_dataset: bool = True):
        """
        Charge le dataset d'entraînement.

        Parameters
        ----------
        return_dataset : bool
            Si False, ne retourne pas le dataset (utilisé uniquement pour le prétraitement).

        Returns
        -------
        MyMICC2000
            Dataset personnalisé contenant les clips vidéo d'entraînement.
        """
        if return_dataset:
            self.train_ids = self.load_train_ids()
            self.train_clips = self.create_clips(
                self.train_dir, self.train_ids,
                clip_length=self.clip_length, stride=self.stride,
                read_target=False
            )
            return MyMICC2000(self.train_clips, self.transform, clip_length=self.clip_length)
        else:
            return

    def get_loaders(self, batch_size: int, shuffle_train: bool = True, pin_memory: bool = False, num_workers: int = 0) -> [DataLoader, DataLoader]:
        """
        Crée et retourne les DataLoaders pour l'entraînement et le test.

        Parameters
        ----------
        batch_size : int
            Taille des lots de données.
        shuffle_train : bool
            Si True, mélange le dataset d'entraînement.
        pin_memory : bool
            Si True, utilise la mémoire verrouillée pour accélérer le transfert vers le GPU.
        num_workers : int
            Nombre de processus de chargement de données.

        Returns
        -------
        tuple(DataLoader, DataLoader)
            Les DataLoaders pour l'entraînement et le test.
        """
        train_loader = DataLoader(
            dataset=self.get_train_data(return_dataset=True),
            batch_size=batch_size,
            shuffle=shuffle_train,
            pin_memory=pin_memory,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            dataset=self.get_test_data(),
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers
        )
        return train_loader, test_loader

    def load_train_ids(self):
        """
        Charge les identifiants des vidéos d'entraînement.

        Returns
        -------
        List[str]
            Liste triée des identifiants de vidéos.
        """
        return sorted([basename(d) for d in glob(join(self.train_dir, '**')) if isdir(d)])

    def create_clips(self, dir_path, ids, clip_length=16, stride=1, read_target=False):
        """
        Génère des clips vidéo à partir des images contenues dans les répertoires.

        Parameters
        ----------
        dir_path : str
            Chemin du dossier contenant les images des vidéos.
        ids : list
            Liste des identifiants des vidéos.
        clip_length : int
            Nombre d'images par clip.
        stride : int
            Pas de déplacement de la fenêtre de clips.
        read_target : bool
            Si True, tente de charger les annotations de test (sinon, ignore).

        Returns
        -------
        numpy.ndarray
            Tableau contenant les chemins des clips créés.
        """
        clips = []
        print(f"Création des clips pour {dir_path} avec une longueur de {clip_length}...")
        for idx in tqdm(ids):
            frames = sorted([x for x in glob(join(dir_path, idx, "*.jpg"))])
            num_frames = len(frames)
            for window in range(0, num_frames - clip_length + 1, stride):
                clips.append(frames[window:window + clip_length])
        return np.array(clips)


# Classe de dataset personnalisé pour MICC_F2000

class MyMICC2000(Dataset):

    def __init__(self, clips, transform=None, clip_length=16):
        """
        Initialise l'ensemble de données MyMICC2000 représentant des clips vidéo.

        Parameters
        ----------
        clips : list
            Liste contenant, pour chaque clip, les chemins des images qui le composent.
        transform : callable, optionnel
            Fonction de transformation à appliquer aux clips.
        clip_length : int, optionnel (par défaut 16)
            Nombre d'images par clip.
        """
        self.clips = clips
        self.transform = transform
        self.shape = (3, clip_length, 256, 512)

    def __len__(self):
        # Ici, la longueur est fixée arbitrairement à 10000 pour simuler un dataset de grande taille
        return 10000

    def __getitem__(self, index):
        """
        Retourne un échantillon du dataset sous forme de clip vidéo.

        Parameters
        ----------
        index : int
            Indice de l'échantillon à récupérer.

        Returns
        -------
        tuple
            (sample, index_) où sample est le clip vidéo (tableau NumPy transformé) et index_ est un indice aléatoire du clip.
        """
        # Génère un index aléatoire parmi les clips disponibles
        index_ = torch.randint(0, len(self.clips), size=(1,)).item()
        # Charge les images du clip correspondant et les empile en un tableau NumPy
        sample = np.stack([np.uint8(io.imread(img_path)) for img_path in self.clips[index_]])
        # Applique la transformation si définie
        sample = self.transform(sample) if self.transform else sample
        return sample, index_


# Fonctions utilitaires et classes de transformations
def get_target_label_idx(labels, targets):
    """
    Retourne les indices des labels présents dans targets.

    Parameters
    ----------
    labels : array
        Tableau des labels.
    targets : list/tuple
        Labels cibles.

    Returns
    -------
    list
        Liste des indices correspondants aux labels cibles.
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Applique la normalisation globale du contraste à un tenseur :
      - Soustrait la moyenne.
      - Normalise par la norme L1 ou L2.

    Parameters
    ----------
    x : torch.tensor
        Tenseur d'entrée (par exemple, une image).
    scale : str, optionnel ('l1' ou 'l2', par défaut 'l2')
        Type de normalisation.

    Returns
    -------
    torch.tensor
        Tenseur normalisé.
    """
    assert scale in ('l1', 'l2')
    n_features = int(np.prod(x.shape))
    mean = torch.mean(x)
    x -= mean
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    elif scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features
    x /= x_scale
    return x


class ToFloatTensor3D(object):
    """
    Convertit une vidéo (tableau NumPy) en un tenseur FloatTensor PyTorch.
    """
    def __init__(self, normalize=True):
        self._normalize = normalize

    def __call__(self, sample):
        """
        Transforme une séquence vidéo du format (T, H, W, C) au format (C, T, H, W)
        et normalise les valeurs si nécessaire.

        Parameters
        ----------
        sample : np.ndarray
            Vidéo sous forme de tableau NumPy.

        Returns
        -------
        torch.Tensor
            Tenseur PyTorch.
        """
        if len(sample) == 3:
            X, Y, _ = sample
        else:
            X = sample
        X = X.transpose(3, 0, 1, 2)
        if self._normalize:
            X = X / 255.
        X = np.float32(X)
        return torch.from_numpy(X)


class ToFloatTensor3DMask(object):
    """
    Convertit une vidéo en un tenseur FloatTensor PyTorch avec gestion d'un masque.
    """
    def __init__(self, normalize=True, has_x_mask=True, has_y_mask=True):
        self._normalize = normalize
        self.has_x_mask = has_x_mask
        self.has_y_mask = has_y_mask

    def __call__(self, sample):
        X = sample
        X = X.transpose(3, 0, 1, 2)
        X = np.float32(X)
        if self._normalize:
            if self.has_x_mask:
                X[:-1] = X[:-1] / 255.
            else:
                X = X / 255.
        return torch.from_numpy(X)


from scipy.ndimage.morphology import binary_dilation

class RemoveBackground:
    """
    Supprime l'arrière-plan d'une image ou vidéo en appliquant un seuil de différence.
    """
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, sample: tuple) -> tuple:
        X, Y, background = sample
        mask = np.uint8(np.sum(np.abs(np.int32(X) - background), axis=-1) > self.threshold)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.stack([binary_dilation(mask_frame, iterations=5) for mask_frame in mask])
        X *= mask
        return X, Y, background
