# Importation des bibliothèques nécessaires
import os
import sys
import time
import logging
import numpy as np
# Importation des modules pour afficher une barre de progression
from tqdm import tqdm
# Bibliothèque PyTorch pour la création et l'entraînement de modèles
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
# Importation pour la visualisation avec TensorBoard
from tensorboardX import SummaryWriter
# Pour les courbes ROC et l'AUC
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import roc_curve, auc

def pretrain(ae_net: nn.Module, train_loader: DataLoader, out_dir: str, tb_writer: SummaryWriter, device: str, ae_learning_rate: float, ae_weight_decay: float, ae_lr_milestones: list, ae_epochs: int, log_frequency: int, batch_accumulation: int, debug: bool) -> str:
    """
    Entraînement du réseau AutoEncoder complet.

    Paramètres :
    - ae_net : nn.Module : Le réseau AutoEncoder
    - train_loader : DataLoader : Chargeur de données pour l'entraînement
    - out_dir : str : Répertoire de sauvegarde des checkpoints
    - tb_writer : SummaryWriter : Objet pour TensorBoard
    - device : str : Dispositif (CPU/GPU) pour l'entraînement
    - ae_learning_rate : float : Taux d'apprentissage pour l'AutoEncoder
    - ae_weight_decay : float : Décroissance du poids
    - ae_lr_milestones : list : Époques pour ajuster le taux d'apprentissage
    - ae_epochs : int : Nombre d'époques d'entraînement
    - log_frequency : int : Fréquence d'enregistrement des logs
    - batch_accumulation : int : Nombre de lots avant d'appliquer les gradients
    - debug : bool : Mode débogage (utiliser 5 premiers lots seulement)

    Retourne :
    - str : Chemin vers le checkpoint du modèle sauvegardé
    """
    
    # Initialisation du logger
    logger = logging.getLogger()
    
    # Passer le modèle à l'état "train" et l'envoyer sur le dispositif (CPU/GPU)
    ae_net = ae_net.train().to(device)

    # Définir l'optimiseur et le scheduler pour le taux d'apprentissage
    optimizer = Adam(ae_net.parameters(), lr=ae_learning_rate, weight_decay=ae_weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=ae_lr_milestones, gamma=0.1)

    # Compteur pour l'accumulation des gradients
    j_ba_steps = 0
    kk = 1  # Compteur pour TensorBoard

    # Boucle d'entraînement sur chaque époque
    for epoch in range(ae_epochs):
        loss_epoch = 0.0  # Perte cumulée pour l'époque
        n_batches = 0  # Nombre de lots traités
        optimizer.zero_grad()  # Réinitialiser les gradients

        # Boucle sur les lots d'entraînement
        for idx, (data, _) in enumerate(tqdm(train_loader, total=len(train_loader), leave=False)):
            
            if debug and idx == 5:
                break  # Mode debug: utiliser seulement 5 lots
            
            data = data.to(device)  # Déplacer les données vers le périphérique

            # Passer les données dans le réseau et calculer la perte de reconstruction
            x_r = ae_net(data)
            scores = torch.sum((x_r - data) ** 2, dim=tuple(range(1, x_r.dim())))
            loss = torch.mean(scores)
            loss.backward()  # Calculer les gradients

            # Accumulation des gradients si spécifié
            j_ba_steps += 1
            if batch_accumulation != -1:
                if j_ba_steps % batch_accumulation == 0:
                    optimizer.step()  # Mettre à jour les poids
                    optimizer.zero_grad()  # Réinitialiser les gradients
                    j_ba_steps = 0  # Réinitialiser le compteur
            else:
                optimizer.step()
                optimizer.zero_grad()

            if np.isnan(loss.item()):  # Vérification des NaN dans la perte
                logger.info("Found nan values in loss")
                sys.exit(0)

            loss_epoch += loss.item()
            n_batches += 1

            # Log périodique
            if idx != 0 and idx % ((len(train_loader)//log_frequency)+1) == 0:
                logger.info(f"PreTrain at epoch: {epoch+1} ([{idx}]/[{len(train_loader)}]) ==> Recon Loss: {loss_epoch/idx:.4f}")
                tb_writer.add_scalar('pretrain/recon_loss', loss_epoch/idx, kk)
                kk += 1

        # Mise à jour du taux d'apprentissage
        scheduler.step()
        if epoch in ae_lr_milestones:
            logger.info(f"LR scheduler: New learning rate is {scheduler.get_lr()[0]}")

        # Sauvegarde du modèle
        ae_net_checkpoint = os.path.join(out_dir, 'best_ae_ckp.pth')
        torch.save({'ae_state_dict': ae_net.state_dict()}, ae_net_checkpoint)
        logger.info(f"Saved best autoencoder at: {ae_net_checkpoint}")

    logger.info("Finished pretraining.")
    return ae_net_checkpoint  # Retourne le chemin du checkpoint

def train(net: nn.Module, train_loader: DataLoader, centers: dict, out_dir: str, tb_writer: SummaryWriter, device: str, learning_rate: float, weight_decay: float, lr_milestones: list, epochs: int, nu: float, boundary: str, batch_accumulation: int, warm_up_n_epochs: int, log_frequency: int, debug: bool) -> str:
    """
    Entraînement du réseau Encoder pour la tâche one-class.

    Paramètres :
    - net : nn.Module : Le réseau Encoder
    - train_loader : DataLoader : Chargeur de données pour l'entraînement
    - centers : dict : Dictionnaire des centres des hypersphères
    - out_dir : str : Répertoire pour sauvegarder les checkpoints
    - tb_writer : SummaryWriter : Objet pour TensorBoard
    - device : str : Périphérique d'entraînement (CPU/GPU)
    - learning_rate : float : Taux d'apprentissage
    - weight_decay : float : Décroissance du poids
    - lr_milestones : list : Époques où ajuster le taux d'apprentissage
    - epochs : int : Nombre d'époques d'entraînement
    - nu : float : Paramètre de compromis
    - boundary : str : Type de frontière ('soft', 'hard')
    - batch_accumulation : int : Nombre de lots avant d'appliquer les gradients
    - warm_up_n_epochs : int : Nombre d'époques pour l'échauffement
    - log_frequency : int : Fréquence des logs
    - debug : bool : Utilisation du mode débogage

    Retourne :
    - str : Chemin du checkpoint du modèle sauvegardé
    """

    logger = logging.getLogger()  # Initialisation du logger

    # Définition de l'optimiseur et du scheduler pour le taux d'apprentissage
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    # Initialisation des rayons des hypersphères
    R = {k: torch.tensor(0.0, device=device) for k in centers.keys()}

    # Démarrage de l'entraînement
    logger.info("Starting training...")
    kk = 1  # Compteur pour TensorBoard
    net.train().to(device)  # Passer en mode entraînement
    best_loss = 1.e6  # Initialiser la meilleure perte à une valeur élevée

    for epoch in range(epochs):
        j = 0  # Compteur pour les lots
        loss_epoch = 0.0  # Perte de l'époque
        n_batches = 0  # Nombre de lots traités
        d_from_c = {}  # Dictionnaire des distances des points aux centres
        optimizer.zero_grad()  # Réinitialiser les gradients

        # Boucle sur les lots de données
        for idx, (data, _) in enumerate(tqdm(train_loader, total=len(train_loader), leave=False)):
            if debug and idx == 5:
                break  # Mode debug: limiter à 5 lots
            
            data = data.to(device)  # Déplacer les données sur le périphérique

            zipped = net(data)  # Passer les données dans le réseau
            
            # Calcul de la perte
            dist, loss = eval_ad_loss(zipped=zipped, c=centers, R=R, nu=nu, boundary=boundary)

            # Mise à jour des distances des points aux centres
            for k in dist.keys():
                if k not in d_from_c:
                    d_from_c[k] = 0
                d_from_c[k] += torch.mean(dist[k]).item()

            # Rétropropagation des gradients
            loss.backward()
            j += 1

            if batch_accumulation != -1:
                if j == batch_accumulation:
                    optimizer.step()
                    optimizer.zero_grad()
                    j = 0
            else:
                optimizer.step()
                optimizer.zero_grad()

            loss_epoch += loss.item()
            n_batches += 1

            if np.isnan(loss.item()):
                logger.info("Found NaN values in loss")
                sys.exit(0)

            # Affichage périodique des logs
            if idx != 0 and idx % ((len(train_loader)//log_frequency)+1) == 0:
                logger.info(f"Training epoch: {epoch+1} ==> Loss: {loss_epoch/idx:.4f}")
                tb_writer.add_scalar('train/total_loss', loss_epoch/idx, kk)
                kk += 1

        scheduler.step()  # Mise à jour du taux d'apprentissage

        # Sauvegarde du modèle
        if loss_epoch / n_batches < best_loss:
            best_loss = loss_epoch / n_batches
            model_checkpoint = os.path.join(out_dir, f"model_best_epoch_{epoch+1}.pth")
            torch.save(net.state_dict(), model_checkpoint)
            logger.info(f"Saved new best model at: {model_checkpoint}")

    return model_checkpoint  # Retourne le chemin du meilleur modèle sauvegardé

# Fonction pour tester le modèle
def test(normal_class: str, is_texture: bool, net: nn.Module, test_loader: DataLoader, R: dict, c: dict, device: str, boundary: str, debug: bool) -> [float, float]:
    """
    Évalue le réseau d'encodeur sur l'ensemble de test et calcule la performance en termes de AUC et d'accuracy équilibrée.

    Paramètres
    ----------
    normal_class : str
        Nom de la classe à tester.
    is_texture : bool
        True si les données d'entrée appartiennent à une classe de type texture.
    net : nn.Module
        Réseau d'encodeur.
    test_loader : DataLoader
        DataLoader pour l'ensemble de test.
    R : dict
        Dictionnaire contenant les valeurs des rayons des hypersphères pour chaque couche.
    c : dict
        Dictionnaire contenant les centres des hypersphères pour chaque couche.
    device : str
        Dispositif sur lequel exécuter le modèle ('cpu' ou 'cuda').
    boundary : str
        Type de frontière utilisé pour le calcul de la distance.
    debug : bool
        Si True, utilise uniquement les 10 premiers lots pour les tests.

    Retours
    -------
    test_auc : float
        AUC (Area Under the ROC Curve).
    balanced_accuracy : float
        Précision équilibrée maximale.
    """
    logger = logging.getLogger()

    # Début du processus de test
    logger.info('Démarrage des tests...')
    
    idx_label_score = []  # Liste pour stocker les labels et les scores pour calculer l'AUC
    
    net.eval().to(device)  # Mise du modèle en mode évaluation

    with torch.no_grad():  # Pas de calcul de gradients durant les tests
        for idx, (data, labels) in enumerate(tqdm(test_loader, total=len(test_loader), desc=f"Test de la classe : {normal_class}", leave=False)):
            # Si en mode debug, ne tester que les 5 premiers lots
            if debug and idx == 5: break

            data = data.to(device)
            
            if is_texture:  # Traitement spécial pour les données de texture (découper les images en patchs)
                _, _, h, w = data.shape
                assert h == w, "Hauteur et Largeur doivent être égales!"  # Vérification que l'image est carrée
                patch_size = 64  # Taille du patch
                
                # Découper l'image en patchs de 64x64
                patches = [
                        data[:, :, h_:h_+patch_size, w_:w_+patch_size]
                        for h_ in range(0, h, patch_size)
                        for w_ in range(0, w, patch_size)
                    ]

                patches = torch.stack(patches, dim=1)  # Empiler les patches en un seul tenseur
                
                # Calculer les scores d'anomalie pour chaque patch
                scores = torch.stack([get_scores(
                                        zipped=net(batch), 
                                        c=c, R=R, 
                                        device=device, 
                                        boundary=boundary, 
                                        is_texture=is_texture) 
                                    for batch in patches
                                ])  # Collecte des scores
            else:  # Si les données ne sont pas de type texture, calculer les scores pour l'image entière
                scores = get_scores(zipped=net(data), c=c, R=R, device=device, boundary=boundary, is_texture=is_texture)

            # Stocker les labels et les scores associés pour calculer l'AUC
            idx_label_score += list(
                                zip(
                                    labels.cpu().data.numpy().tolist(),  # Labels du lot actuel
                                    scores.cpu().data.numpy().tolist()  # Scores d'anomalie pour le lot actuel
                                )
                            )

    # Calculer l'AUC et la précision équilibrée
    labels, scores = zip(*idx_label_score)  # Séparer les labels et les scores
    labels = np.array(labels)
    scores = np.array(scores)

    # Calculer la courbe ROC et la précision équilibrée
    fpr, tpr, _ = roc_curve(labels, scores)  # Taux de faux positifs et de vrais positifs
    balanced_accuracy = np.max((tpr + (1 - fpr)) / 2)  # Précision équilibrée maximale
    auroc = auc(fpr, tpr)  # Calcul de l'AUC (Area Under the ROC Curve)
    
    # Log des résultats du test
    logger.info(f"Résultats des tests ===> AUC : {auroc:.4f} --- maxB : {balanced_accuracy:.4f}")
    logger.info('Tests terminés!')

    return auroc, balanced_accuracy

# Fonction pour calculer la perte d'anomalie
def eval_ad_loss(zipped: dict, c: dict, R: dict, nu: float, boundary: str) -> [dict, torch.Tensor]:
    """
    Évalue la perte pour l'encodeur dans un cadre de classification one-class.

    Paramètres
    ----------
    zipped : dict
        Dictionnaire contenant les caractéristiques de sortie du modèle.
    c : dict
        Dictionnaire des centres des hypersphères à chaque couche.
    R : dict 
        Dictionnaire des rayons des hypersphères à chaque couche.
    nu : float
        Paramètre de régulation pour ajuster la perte.
    boundary: str
        Type de frontière ('soft' ou autre).

    Retours
    -------
    dist : dict
        Dictionnaire des distances entre les caractéristiques et les centres des hypersphères à chaque couche.
    loss : torch.Tensor
        Valeur totale de la perte calculée.
    """
    dist = {}  # Dictionnaire pour stocker les distances
    loss = 1  # Initialisation de la perte

    # Boucle sur chaque couche pour calculer la distance et la perte
    for (k, v) in zipped.items():
        # Calcul de la distance entre les caractéristiques et le centre de l'hypersphère
        dist[k] = torch.sum((v - c[k].unsqueeze(0)) ** 2, dim=1)
        
        # Si la frontière est 'soft', calculer la perte basée sur la distance et le rayon
        if boundary == 'soft':
            scores = dist[k] - R[k] ** 2  # Calcul des scores
            # Ajouter la perte avec un ajustement via le paramètre nu
            loss += R[k] ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            # Si la frontière n'est pas 'soft', utiliser la moyenne des distances
            loss += torch.mean(dist[k])

    return dist, loss  # Retourner les distances et la perte totale

# Fonction pour obtenir les scores d'anomalie
def get_scores(zipped: dict, c: dict, R: dict, device: str, boundary: str, is_texture: bool) -> float:
    """
    Évalue le score d'anomalie en fonction des caractéristiques extraites par l'encodeur.

    Paramètres
    ----------
    zipped : dict
        Dictionnaire contenant les caractéristiques de sortie.
    c : dict
        Dictionnaire des centres des hypersphères à chaque couche.
    R : dict 
        Dictionnaire des rayons des hypersphères à chaque couche.
    device : str
        Dispositif sur lequel exécuter le calcul ('cuda' ou 'cpu').
    boundary: str
        Type de frontière ('soft' ou autre).
    is_texture : bool
        Indique si les données appartiennent à une classe de type texture.

    Retours
    -------
    scores : float
        Score d'anomalie pour chaque image.
    """
    # Calcul des distances entre les caractéristiques et les centres des hypersphères
    dist = {item[0]: torch.norm(item[1] - c[item[0]].unsqueeze(0), dim=1) for item in zipped.items()}
    
    # Initialisation du tableau des scores
    shape = dist[list(dist.keys())[0]].shape[0]
    scores = torch.zeros((shape,), device=device)
    
    # Calcul du score d'anomalie en fonction de la frontière
    for k in dist.keys():
        if boundary == 'soft':
            scores += dist[k] - R[k]  # Calcul des scores avec la frontière soft
        else:
            scores += dist[k]  # Simplement additionner les distances si frontière non 'soft'
    
    # Retourner le score maximal normalisé pour les textures ou la moyenne des scores pour les autres types
    return scores.max() / len(list(dist.keys())) if is_texture else scores / len(list(dist.keys()))
