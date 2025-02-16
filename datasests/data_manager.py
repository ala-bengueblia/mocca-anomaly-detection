import os  # Module pour interagir avec le système de fichiers (ex. vérifier si un fichier existe)
import sys  # Module qui fournit des fonctionnalités spécifiques à l'interpréteur Python
import logging  # Module pour gérer les logs et afficher des messages d'information ou d'erreur
from .mvtec import MVTec_DataHolder  # Classe permettant de gérer le dataset MVTec Anomaly
from .cifar10 import CIFAR10_DataHolder  # Classe pour gérer le dataset CIFAR-10
from .shanghaitech import ShanghaiTech_DataHolder  # Classe pour gérer le dataset ShanghaiTech


# Liste des datasets supportés par le DataManager
AVAILABLE_DATASETS = ('casia1','casia2' 'micc_f220', 'micc_f2000') 

class DataManager:
    """
    Classe pour gérer et charger les données.
    """
    def __init__(self, dataset_name: str, data_path: str, normal_class: int, clip_length: int = 16, only_test: bool = False):
        """
        Initialise le DataManager pour le dataset donné.

        Parameters
        ----------
        dataset_name : str
            Nom du dataset à utiliser.
        data_path : str 
            Chemin vers les données du dataset.
        normal_class : int 
            Classe considérée comme normale.
        clip_length : int 
            Nombre de frames vidéo par clip (utilisé uniquement pour ShanghaiTech).
        only_test : bool
            Indique si le modèle est en mode test.
        """
        self.dataset_name = dataset_name  # Nom du dataset
        self.data_path = data_path  # Chemin des données
        self.normal_class = normal_class  # Classe normale
        self.clip_length = clip_length  # Longueur des clips vidéo (ShanghaiTech seulement)
        self.only_test = only_test  # Mode test activé ou non

        # Vérifie la disponibilité des données immédiatement après l'initialisation
        self.__check_dataset()

    def __check_dataset(self) -> None:
        """
        Vérifie si le dataset requis est disponible.

        Raises
        ------
        AssertionError
            Si le dataset n'est pas trouvé ou si le chemin est incorrect.
        """
        assert self.dataset_name in AVAILABLE_DATASETS, f"Le dataset {self.dataset_name} n'est pas disponible."
        assert os.path.exists(self.data_path), f"Le dataset {self.dataset_name} est disponible, mais introuvable à : \n{self.data_path}"

    def get_data_holder(self):
        """
        Renvoie le gestionnaire de données correspondant au dataset requis.

        Returns
        -------
        MVTec_DataHolder | CIFAR10_DataHolder | ShanghaiTech_DataHolder
            Gestionnaire de données du dataset requis.
        """
        if self.dataset_name == 'cifar10':
            return CIFAR10_DataHolder(root=self.data_path, normal_class=self.normal_class)

        if self.dataset_name == 'ShanghaiTech':
            return ShanghaiTech_DataHolder(root=self.data_path, clip_length=self.clip_length)

        if self.dataset_name == 'MVTec_Anomaly':
            texture_classes = ("carpet", "grid", "leather", "tile", "wood")
            object_classes = ("bottle", "hazelnut", "metal_nut", "screw")
            
            # Vérifie si la classe sélectionnée est de type texture
            is_texture = self.normal_class in texture_classes
            if is_texture:
                image_size = 512  # Taille des images pour les textures
                patch_size = 64  # Taille du patch pour analyse locale
                rotation_range = (0, 45)  # Plage de rotation de 0° à 45° pour les textures
            else:
                patch_size = 1  # Pas de division en patches pour les objets
                image_size = 128  # Taille des images pour les objets
                rotation_range = (-45, 45) if self.normal_class in object_classes else (0, 0)

            return MVTec_DataHolder(
                data_path=self.data_path,
                category=self.normal_class, 
                image_size=image_size, 
                patch_size=patch_size, 
                rotation_range=rotation_range, 
                is_texture=is_texture
            )
