import numpy as np  # Pour les calculs numériques
from PIL import Image  # Pour la gestion des images
import torch  # La bibliothèque PyTorch pour le deep learning
from torch.utils.data import DataLoader, Subset  # Pour charger des données par lots et créer des sous-ensembles
from torchvision.datasets import ImageFolder  # Dataset de PyTorch pour charger des images d'un répertoire
import torchvision.transforms as transforms  # Pour appliquer des transformations aux images


def obtenir_indices_labels_cibles(labels: np.array, cibles: np.array):
    """
    Obtenir les indices des labels présents dans les cibles.
    
    Paramètres
    ----------
    labels : np.array
        Tableau des labels
    cibles : np.array
        Tableau des labels cibles
    
    Retourne
    -------
    Liste des indices des labels cibles
    
    """
    return np.argwhere(np.isin(labels, cibles)).flatten().tolist()

def normalisation_contraste_globale(x: torch.tensor, echelle: str='l1') -> torch.Tensor:
    """Applique une normalisation de contraste global sur le tenseur, c'est-à-dire soustrait la moyenne des caractéristiques (pixels) et normalise selon une échelle,
    qui peut être soit l'écart type, la norme L1 ou L2 des caractéristiques (pixels).
    Remarque : il s'agit d'une normalisation *par échantillon* au niveau global des caractéristiques (et non au niveau du dataset).

    Paramètres
    ----------
    x : torch.tensor
        Échantillon de données
    echelle : str
        Échelle à utiliser

    Retourne
    -------
    Tenseur normalisé des caractéristiques

    """
    assert echelle in ('l1', 'l2')  # Vérifie que l'échelle est soit 'l1', soit 'l2'

    n_caracteristiques = int(np.prod(x.shape))  # Calcul du nombre total de caractéristiques (pixels)

    # Calcul de la moyenne sur toutes les caractéristiques (pixels) par échantillon
    moyenne = torch.mean(x)  # Calcule la moyenne
    x -= moyenne  # Soustrait la moyenne

    # Calcul de l'échelle (L1 ou L2)
    x_echelle = torch.mean(torch.abs(x)) if echelle == 'l1' else torch.sqrt(torch.sum(x ** 2)) / n_caracteristiques

    return x / x_echelle  # Normalisation du tenseur

class GestionnaireDeDonneesCASIA2(object):
    """Classe de gestion des données CASIA2"""
    
    def __init__(self, chemin: str, classe_normale=0):
        """Initialisation du gestionnaire de données CASIA2
        
        Paramètres
        ----------
        chemin : str
            Chemin vers le dossier contenant les données
        classe_normale : int
            Index de la classe normale

        """
        self.chemin = chemin  # Chemin vers les données

        # Nombre total de classes = 2, i.e., 0: normal, 1: anomalies
        self.n_classes = 2  # Nombre de classes (0: normal, 1: anomalie)
        
        # Tuple contenant la classe normale
        self.classes_normales = tuple([classe_normale])  # Classe normale
        
        # Liste des classes anormales
        self.classes_anormales = list(range(0, 10))  # Classes anormales
        self.classes_anormales.remove(classe_normale)  # Retire la classe normale de la liste des anomalies

        # Initialisation des jeux de données
        self.__initialiser_donnees_train_test(classe_normale)  # Initialise les datasets d'entraînement et de test

    def __initialiser_donnees_train_test(self, classe_normale: int) -> None:
        """Initialiser les jeux de données.
        
        Paramètres
        ----------
        classe_normale : int
            L'index de la classe non-anormale
        
        """
        # Définition des transformations pour les images
        self.transformation = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: normalisation_contraste_globale(x, echelle='l1'))
                                    ])

        # Définition des transformations pour les labels, i.e., on met à 0 tous les labels appartenant aux classes anormales
        self.transformation_target = transforms.Lambda(lambda x: int(x in self.classes_anormales))

        # Initialisation du jeu de données d'entraînement
        self.train_set = ImageFolder(
                                root=self.chemin + '/train', 
                                transform=self.transformation, 
                                target_transform=self.transformation_target
                            )

        # Sous-ensemble du jeu d'entraînement ne contenant que les images de la classe normale
        train_idx_normal = obtenir_indices_labels_cibles(labels=np.array([s[1] for s in self.train_set.samples]), cibles=self.classes_normales)
        self.train_set = Subset(self.train_set, train_idx_normal)

        # Initialisation du jeu de données de test
        self.test_set = ImageFolder(
                                root=self.chemin + '/test', 
                                transform=self.transformation, 
                                target_transform=self.transformation_target
                            )

    def obtenir_chargeurs(self, taille_batch: int, shuffle_train: bool=True, pin_memory: bool=False, nb_workers: int = 0) -> [torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Retourne les chargeurs de données CASIA2

        Paramètres
        ----------
        taille_batch : int
            Taille du batch
        shuffle_train : bool
            Si True, les données d'entraînement sont mélangées
        pin_memory : bool
            Si True, la mémoire est verrouillée
        nb_workers : int 
            Nombre de travailleurs pour le DataLoader
        
        Retourne
        -------
        chargeurs : DataLoader
            Chargeurs de données pour l'entraînement et le test

        """
        chargeur_train = DataLoader(
                            dataset=self.train_set, 
                            batch_size=taille_batch, 
                            shuffle=shuffle_train,
                            pin_memory=pin_memory,
                            num_workers=nb_workers
                        )
        chargeur_test = DataLoader(
                            dataset=self.test_set, 
                            batch_size=taille_batch,
                            pin_memory=pin_memory,
                            num_workers=nb_workers
                        )
        return chargeur_train, chargeur_test

from torchvision.datasets import VisionDataset
from PIL import Image

class MyCASIA2(VisionDataset):
    """
    Subclass of torchvision VisionDataset to modify the __getitem__ method.
    
    This version also returns the index of the data sample along with the image and target.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the MyCASIA2 dataset by calling the parent VisionDataset constructor.
        """
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        """
        Retrieve the image, target, and the index of a sample from the dataset.

        Args:
            index (int): Index of the data sample.

        Returns:
            tuple: (image, target, index) where target is the class label.
        """
        img, target = self.data[index], self.targets[index]

        # Convert the image to PIL format
        img = Image.fromarray(img)

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)
        
        return img, target, index
