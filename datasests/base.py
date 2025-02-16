from abc import ABC, abstractmethod  # Pour définir des classes et méthodes abstraites
import torch  # La bibliothèque PyTorch pour le deep learning
import numpy as np  # Pour les calculs numériques
from torch.utils.data import Dataset  # Classe de base pour les datasets dans PyTorch


class DatasetBase(Dataset, ABC):
    """
    Classe de base pour tous les datasets.
    """
    @abstractmethod
    def test(self, *args):
        """
        Définit le dataset en mode test.
        """
        pass

    @property
    @abstractmethod
    def shape(self):
        """
        Retourne la forme des exemples du dataset.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Retourne le nombre d'exemples.
        """
        pass

    @abstractmethod
    def __getitem__(self, i):
        """
        Retourne l'exemple à l'index i.
        """
        pass


class OneClassDataset(DatasetBase):
    """
    Classe de base pour tous les datasets de classification one-class.
    """
    @abstractmethod
    def val(self, *args):
        """
        Définit le dataset en mode validation.
        """
        pass

    @property
    @abstractmethod
    def test_classes(self):
        """
        Retourne toutes les classes de test possibles.
        """
        pass


class VideoAnomalyDetectionDataset(DatasetBase):
    """
    Classe de base pour tous les datasets de détection d'anomalies vidéo.
    """
    @property
    @abstractmethod
    def test_videos(self):
        """
        Retourne tous les identifiants des vidéos de test.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Retourne le nombre d'exemples.
        """
        pass

    @property
    def raw_shape(self):
        """
        Solution de contournement pour la forme brute des données.
        """
        return self.shape

    @abstractmethod
    def __getitem__(self, i):
        """
        Retourne l'exemple à l'index i.
        """
        pass

    @abstractmethod
    def load_test_sequence_gt(self, video_id):
        """
        Charge la vérité terrain d'une vidéo de test en mémoire.

        :param video_id: l'identifiant de la vidéo de test à charger.
        :return: la vérité terrain sous forme de np.ndarray (n_frames,)
        """
        pass

    @property
    @abstractmethod
    def collate_fn(self):
        """
        Retourne une fonction qui décide comment fusionner une liste d'exemples en un batch.
        """
        pass


class ToFloatTensor3D(object):
    """ Convertit les vidéos en FloatTensors """
    def __init__(self, normalize=True):
        self._normalize = normalize 

    def __call__(self, sample):
        if len(sample) == 3:
            X, Y, _ = sample
        else:
            X = sample

        # Permute les axes de couleur car numpy image : T x H x W x C
        X = X.transpose(3, 0, 1, 2)  # Transforme en C x T x H x W

        if self._normalize:
            X = X / 255.  # Normalise les valeurs entre 0 et 1

        X = np.float32(X)  # Convertit en float32
        return torch.from_numpy(X)  # Convertit en tensor PyTorch


class ToFloatTensor3DMask(object):
    """ Convertit les vidéos en FloatTensors avec un masque """
    def __init__(self, normalize=True, has_x_mask=True, has_y_mask=True):
        self._normalize = normalize  # Indique si la normalisation est activée
        self.has_x_mask = has_x_mask  # Indique si X a un masque
        self.has_y_mask = has_y_mask  # Indique si Y a un masque

    def __call__(self, sample):
        X = sample
        # Permute les axes de couleur
        X = X.transpose(3, 0, 1, 2)  # Transforme en C x T x H x W

        X = np.float32(X)  # Convertit en float32

        if self._normalize:
            if self.has_x_mask:
                X[:-1] = X[:-1] / 255.  # Normalise tout sauf le masque
            else:
                X = X / 255.  # Normalise tout

        return torch.from_numpy(X)  # Convertit en tensor PyTorch


class RemoveBackground:
    """
    Supprime l'arrière-plan des vidéos en utilisant un seuil.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, sample: tuple) -> tuple:
        X, Y, background = sample

        # Création du masque en fonction du seuil
        mask = np.uint8(np.sum(np.abs(np.int32(X) - background), axis=-1) > self.threshold)
        mask = np.expand_dims(mask, axis=-1)

        # Applique une dilatation binaire sur le masque
        mask = np.stack([binary_dilation(mask_frame, iterations=5) for mask_frame in mask])

        # Applique le masque à X
        X *= mask

        return X, Y, background
