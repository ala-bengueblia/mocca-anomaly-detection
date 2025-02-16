import os  # Pour interagir avec le système de fichiers (créer des dossiers, etc.)
import sys  # Pour gérer les arguments de ligne de commande et quitter le programme si nécessaire
import random  # Pour générer des nombres aléatoires
import logging  # Pour enregistrer les messages de débogage et d'information
import argparse  # Pour parser les arguments passés en ligne de commande
import numpy as np  # Pour les calculs numériques
from tqdm import tqdm  # Pour afficher des barres de progression
import torch  # La bibliothèque PyTorch pour le deep learning
from trainer_svdd import test  # Fonction pour tester le modèle Deep SVDD
from datasets.main import load_dataset  # Fonction pour charger le dataset
from models.deep_svdd.deep_svdd_mnist import MNIST_LeNet_Autoencoder, MNIST_LeNet  # Modèles pour MNIST
from models.deep_svdd.deep_svdd_cifar10 import CIFAR10_LeNet_Autoencoder, CIFAR10_LeNet  # Modèles pour CIFAR-10


parser = argparse.ArgumentParser('AD')# Crée un parseur d'arguments
## General config
parser.add_argument('--n_jobs_dataloader', type=int, default=0, help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
## Model config
parser.add_argument('-zl', '--code-length', default=32, type=int, help='Code length (default: 32)') # Longueur du code latent
parser.add_argument('-ck', '--model-ckp', help='Model checkpoint')# Chemin du checkpoint du modèle
## Data
parser.add_argument('-ax', '--aux-data-filename', default='/media/fabiovalerio/datasets/ti_500K_pseudo_labeled.pickle', help='Path to unlabelled data')# Chemin des données non labellisées
parser.add_argument('-dn', '--dataset-name', choices=('mnist', 'cifar10'), default='mnist', help='Dataset (default: mnist)') # Nom du dataset
parser.add_argument('-ul', '--unlabelled-data', action="store_true", help='Use unlabelled data (default: False)')# Utiliser des données non labellisées
parser.add_argument('-aux', '--unl-data-path', default="/media/fabiovalerio/datasets/ti_500K_pseudo_labeled.pickle", help='Path to unalbelled data')#Chemin des données non labellisée
## Training config
parser.add_argument('-bs', '--batch-size', type=int, default=200, help='Batch size (default: 200)')# Taille du batch
parser.add_argument('-bd', '--boundary', choices=("hard", "soft"), default="soft", help='Boundary (default: soft)')# Type de frontière
parser.add_argument('-ile', '--idx-list-enc', type=int, nargs='+', default=[], help='List of indices of model encoder')# Liste des indices des couches de l'encodeur
args = parser.parse_args()# Parse les arguments


# Get data base path
# Obtient le chemin des données en fonction de l'utilisateur
_user = os.environ['USER']
if _user == 'fabiovalerio':
    data_path = '/media/fabiovalerio/datasets'
elif _user == 'fabiom':
    data_path = '/mnt/datone/datasets/'
else:
    # Lève une erreur si l'utilisateur n'est pas configuré
    raise NotImplementedError('Username %s not configured' % _user)

def main():
    # Vérifie si CUDA est disponible et définit le device
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')

    # Extrait le type de frontière et la classe normale du nom du fichier de checkpoint
    boundary = args.model_ckp.split('/')[-1].split('-')[-3].split('_')[-1]
    normal_class = int(args.model_ckp.split('/')[-1].split('-')[2].split('_')[-1])
    
    # Si la liste des indices des couches de l'encodeur est vide, l'extrait du nom du fichier de checkpoint
    if len(args.idx_list_enc) == 0:
        idx_list_enc = [int(i) for i in args.model_ckp.split('/')[-1].split('-')[-1].split('_')[-1].split('.')]
    else:
        idx_list_enc = args.idx_list_enc
    
    # LOAD DATA
    dataset = load_dataset(args.dataset_name, data_path, normal_class, args.unlabelled_data, args.unl_data_path)

    print(
        # Affiche les paramètres de test
        f"Start test with params"
        f"\n\t\t\t\tCode length    : {args.code_length}"
        f"\n\t\t\t\tEnc layer list : {idx_list_enc}"
        f"\n\t\t\t\tBoundary       : {boundary}"
        f"\n\t\t\t\tNormal class   : {normal_class}"
    )

    # Initialise une liste pour stocker les AUC de test
    test_auc = []
    main_model_ckp_dir = args.model_ckp

    # Parcourt tous les checkpoints dans le répertoire spécifié
    for m_ckp in tqdm(os.listdir(main_model_ckp_dir), total=len(os.listdir(main_model_ckp_dir)), leave=False):
        net_cehckpoint = os.path.join(main_model_ckp_dir, m_ckp)

        # Load model
        # Charge le modèle en fonction du dataset
        net = MNIST_LeNet(args.code_length) if args.dataset_name == 'mnist' else CIFAR10_LeNet(args.code_length)
        st_dict = torch.load(net_cehckpoint)# Charge le checkpoint
        net.load_state_dict(st_dict['net_state_dict'])# Charge les poids du modèle
        
        # TEST
        test_auc_ = test(net, dataset, st_dict['R'], st_dict['c'], device, idx_list_enc, boundary, args)
        del net, st_dict  #Supprime le modèle et le state_dict pour libérer de la mémoire

        test_auc.append(test_auc_) # Ajoute l'AUC de test à la liste
   
    # Calcule la moyenne et l'écart-type des AUC de test
    test_auc = np.asarray(test_auc)
    test_auc_m, test_auc_s = test_auc.mean(), test_auc.std()
    
    # Affiche les résultats
    print("[")
    for tau in test_auc:
        print(tau, ", ")
    print("]")
    print(test_auc)
    print(f"{test_auc_m:.2f} $\pm$ {test_auc_s:.2f}")# Affiche la moyenne et l'écart-type


if __name__ == '__main__':
    main()# Appelle la fonction principale
