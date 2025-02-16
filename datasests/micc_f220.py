import os            # Pour la gestion des fichiers et dossiers
import sys           # Pour les interactions avec l'interpréteur Python
import math          # Pour les opérations mathématiques
import random        # Pour la génération de nombres aléatoires
import numpy as np   # Pour les calculs numériques sur tableaux multidimensionnels
from tqdm import tqdm  # Pour afficher une barre de progression
from PIL import Image  # Pour manipuler les images
from os.path import join  # Pour construire des chemins de fichiers
import torch         # Pour les calculs tensoriels et l'IA avec PyTorch
import torch.nn as nn  # Pour définir des modèles de réseaux de neurones
from torch.utils.data import Dataset, DataLoader, TensorDataset  # Pour gérer les ensembles de données
import torchvision.transforms as T  # Pour appliquer des transformations sur les images
from torchvision.datasets import ImageFolder  # Pour charger des images organisées en dossiers


class MICC_F220_Dataset(ImageFolder):
    """
    Dataset MICC_F220 basé sur ImageFolder avec une modification de la méthode __getitem__
    afin d'ajuster la cible selon la tâche.
    """
    def __init__(self, root: str, transform):
        super(MICC_F220_Dataset, self).__init__(root=root, transform=transform)
        # Indice de la classe correspondant au dossier nommé 'good'
        self.normal_class_idx = self.class_to_idx['good']

    def __getitem__(self, index: int):
        # Récupère le chemin de l'image et son label
        path, target = self.samples[index]

        def read_image(image_path):
            """
            Lit l'image depuis le chemin spécifié et la convertit en RGB.
            """
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')

        # Convertit le label : 0 pour normal, 1 pour anormal
        target = 0 if target == self.normal_class_idx else 1
        # Applique la transformation à l'image
        image = self.transform(read_image(path))
        return image, target


class CustomTensorDataset(TensorDataset):
    """
    Dataset personnalisé pour les images prétraitées stockées sous forme de tenseurs.
    """
    def __init__(self, root: str):
        """
        Charge les données depuis un fichier .npy et initialise le TensorDataset.
        
        Parameters
        ----------
        root : str
            Chemin vers le fichier de données prétraitées.
        """
        self.data = torch.from_numpy(np.load(root))
        super(CustomTensorDataset, self).__init__(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # Retourne l'échantillon et une étiquette par défaut (0)
        return self.data[index], 0


class MICC_F220_DataHolder(object):
    """
    Classe de gestion des données pour le dataset MICC_F220.
    """
    def __init__(self, data_path: str, category: str, image_size: int, patch_size: int, rotation_range: tuple, is_texture: bool):
        """
        Initialise le DataHolder pour MICC_F220.
        
        Parameters
        ----------
        data_path : str
            Chemin vers le dossier contenant les données.
        category : str
            Catégorie des images (par exemple "bottle", "carpet", etc.).
        image_size : int
            Taille en pixels des côtés des images d'entrée.
        patch_size : int
            Taille des patches (pour les textures).
        rotation_range : tuple
            Plage (min, max) d'angles pour la rotation des images.
        is_texture : bool
            True si la catégorie est de type texture.
        """
        self.data_path = data_path
        self.category = category
        self.image_size = image_size
        self.patch_size = patch_size
        self.rotation_range = rotation_range
        self.is_texture = is_texture

    def get_test_data(self) -> Dataset:
        """
        Charge et retourne le dataset de test.
        
        Returns
        -------
        MICC_F220_Dataset
            Dataset personnalisé pour les données de test MICC_F220.
        """
        return MICC_F220_Dataset(
            root=join(self.data_path, f'{self.category}/test'),
            transform=T.Compose([
                T.Resize(self.image_size, interpolation=Image.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        )

    def get_train_data(self, return_dataset: bool = True):
        """
        Charge et prétraite le dataset d'entraînement.
        
        Parameters
        ----------
        return_dataset : bool, optional
            Si True, retourne le dataset prétraité ; sinon, se contente de le prétraiter.
        
        Returns
        -------
        CustomTensorDataset or None
            Le dataset prétraité sous forme de CustomTensorDataset, ou None si return_dataset est False.
        """
        train_data_dir = join(self.data_path, f'{self.category}/train/')
        # Répertoire de cache pour les données prétraitées
        cache_main_dir = join(self.data_path, f'processed/{self.category}')
        os.makedirs(cache_main_dir, exist_ok=True)
        cache_file = join(
            cache_main_dir,
            f'{self.category}_train_dataset_i-{self.image_size}_p-{self.patch_size}_r-{self.rotation_range[0]}--{self.rotation_range[1]}.npy'
        )

        # Si le fichier de cache n'existe pas, le créer
        if not os.path.exists(cache_file):
            def augmentation():
                """
                Retourne les transformations à appliquer aux images.
                """
                if self.is_texture:
                    return T.Compose([
                        T.Resize(self.image_size, interpolation=Image.BILINEAR),
                        T.Pad(padding=self.image_size // 4, padding_mode="reflect"),
                        T.RandomRotation((self.rotation_range[0], self.rotation_range[1])),
                        T.CenterCrop(self.image_size),
                        T.RandomCrop(self.patch_size),
                        T.ToTensor(),
                        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])
                else:
                    return T.Compose([
                        T.Resize(self.image_size, interpolation=Image.BILINEAR),
                        T.Pad(padding=self.image_size // 4, padding_mode="reflect"),
                        T.RandomRotation((self.rotation_range[0], self.rotation_range[1])),
                        T.CenterCrop(self.image_size),
                        T.ToTensor(),
                        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])

            # Charge le dataset d'entraînement avec augmentation
            train_dataset = ImageFolder(root=train_data_dir, transform=augmentation())
            print(f"Création du cache pour le dataset :\n{cache_file}")
            # Calcule le nombre d'époques pour simuler un dataset plus étendu
            nb_epochs = 50000 // len(train_dataset.imgs)
            data_loader = DataLoader(dataset=train_dataset, batch_size=1024, pin_memory=True)

            cache_np = []
            for epoch in tqdm(range(nb_epochs), total=nb_epochs, desc=f"Création du cache pour: {self.category}"):
                for batch_data, _ in tqdm(data_loader, total=len(data_loader), desc=f'Cache epoch: {epoch+1}/{nb_epochs}', leave=False):
                    cache_np.append(batch_data.numpy())
            cache_np = np.vstack(cache_np)
            np.save(cache_file, cache_np)
            print(f"Les images prétraitées ont été sauvegardées ici :\n{cache_file}")

        if return_dataset:
            print(f"Chargement du dataset depuis le cache :\n{cache_file}")
            return CustomTensorDataset(cache_file)
        else:
            return None

    def get_loaders(self, batch_size: int, shuffle_train: bool = True, pin_memory: bool = False, num_workers: int = 0) -> [DataLoader, DataLoader]:
        """
        Retourne les DataLoaders pour l'entraînement et le test.
        
        Parameters
        ----------
        batch_size : int
            Taille du batch.
        shuffle_train : bool, optional
            Si True, mélange le dataset d'entraînement.
        pin_memory : bool, optional
            Si True, utilise la mémoire "pinned" pour accélérer le transfert vers le GPU.
        num_workers : int, optional
            Nombre de threads pour le chargement des données.
        
        Returns
        -------
        list[DataLoader, DataLoader]
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


if __name__ == '__main__':
    """
    Ce bloc permet de prétraiter les images d'entraînement et de les sauvegarder sous forme de fichier .npy.
    """
    textures = ('carpet', 'grid', 'leather', 'tile', 'wood')
    objects_1 = ('bottle', 'hazelnut', 'metal_nut', 'screw')
    objects_2 = ('capsule', 'toothbrush', 'cable', 'pill', 'transistor', 'zipper')
    classes = list(textures) + list(objects_1) + list(objects_2)
    
    for category in classes:
        if category in textures:
            args = dict(
                category=category,
                image_size=512,
                patch_size=64,
                rotation_range=(0, 45),
                is_texture=True
            )
        elif category in objects_1:
            args = dict(
                category=category,
                image_size=128,
                patch_size=-1,
                rotation_range=(-45, 45),
                is_texture=True
            )
        else:
            args = dict(
                category=category,
                image_size=128,
                patch_size=-1,
                rotation_range=(0, 0),
                is_texture=False
            )
    # Lancement du prétraitement pour la dernière catégorie définie dans la boucle
    MICC_F220_DataHolder(data_path="votre/chemin/vers/donnees", **args).get_train_data(return_dataset=False)


