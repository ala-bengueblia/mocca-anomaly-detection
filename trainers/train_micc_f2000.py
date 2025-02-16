import os  # Module pour interagir avec le système d'exploitation (ex : gestion des fichiers et répertoires)
import time  # Module pour mesurer le temps et effectuer des opérations basées sur le temps
import logging  # Module pour la gestion des logs (affichage des messages d'information, erreurs, etc.)
import itertools  # Module pour créer et manipuler des itérations (combinations, permutations, etc.)
import numpy as np  # Bibliothèque pour le calcul scientifique (matrices, algèbre linéaire, etc.)
from tqdm import tqdm  # Module pour afficher des barres de progression lors de l'exécution de boucles
import torch  # Framework pour le calcul tensoriel et l'apprentissage automatique
import torch.nn as nn  # Module pour construire des réseaux de neurones en PyTorch
import torch.nn.functional as F  # Fonctions utiles pour les réseaux de neurones (activation, perte, etc.)
from torch.nn import DataParallel  # Pour l'entraînement sur plusieurs GPU (parallélisme)
from torch.optim import Adam, SGD  # Optimiseurs courants pour l'entraînement de modèles
from torch.optim.lr_scheduler import MultiStepLR  # Pour ajuster dynamiquement le taux d'apprentissage pendant l'entraînement
from torch.utils.data.dataloader import DataLoader  # Pour charger et traiter les données par lots
from sklearn.metrics import roc_auc_score  # Métrique pour évaluer les modèles (AUC-ROC, score de qualité de classification)


# Fonction pour le pré-entraînement de l'autoencodeur
def pretrain(ae_net, train_loader, out_dir, tb_writer, device, args):
    """
    Effectue le pré-entraînement de l'autoencodeur (AE).
    """
    # Initialisation du logger pour suivre les étapes de l'entraînement
    logger = logging.getLogger()
    ae_net = ae_net.train().to(device)  # Mettre le modèle en mode entraînement et sur le bon appareil

    # Choix de l'optimiseur (Adam ou SGD)
    optimizer = Adam(ae_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) if args.optimizer == 'adam' else SGD(ae_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    # Planificateur de taux d'apprentissage qui réduit le taux après certaines époques
    scheduler = MultiStepLR(optimizer, milestones=args.ae_lr_milestones, gamma=0.1)

    ae_epochs = 1 if args.debug else args.ae_epochs  # Nombre d'époques à entraîner (1 si debug)
    logger.info("Début du pré-entraînement de l'autoencodeur...")

    # Boucle d'entraînement
    for epoch in range(ae_epochs):
        recon_loss = 0.0  # Perte de reconstruction
        n_batches = 0  # Compteur de lots

        # Boucle sur les lots d'entraînement
        for idx, (data, _) in enumerate(tqdm(train_loader), 1):
            if args.debug and idx == 2: break  # Arrêter après 2 lots si en mode debug

            data = data.to(device)  # Déplacer les données sur le bon appareil
            optimizer.zero_grad()  # Réinitialiser les gradients
            x_r = ae_net(data)[0]  # Reconstruction des données par l'autoencodeur
            recon_loss_ = torch.mean(torch.sum((x_r - data) ** 2, dim=tuple(range(1, x_r.dim()))))  # Calcul de la perte de reconstruction
            recon_loss_.backward()  # Calcul des gradients
            optimizer.step()  # Mise à jour des poids du modèle

            recon_loss += recon_loss_.item()  # Accumuler la perte
            n_batches += 1  # Incrémenter le compteur de lots

            # Affichage des informations sur la perte toutes les `log_frequency` itérations
            if idx % (len(train_loader) // args.log_frequency) == 0:
                logger.info(f"Pré-entraînement à l'époque: {epoch + 1} [{idx}]/[{len(train_loader)}] ==> Perte de reconstruction: {recon_loss / idx:.4f}")
                tb_writer.add_scalar('pretrain/recon_loss', recon_loss / idx)  # Enregistrer la perte dans TensorBoard

        scheduler.step()  # Mise à jour du taux d'apprentissage

        # Sauvegarder le modèle après chaque époque
        ae_net_checkpoint = os.path.join(out_dir, f'ae_ckp_epoch_{epoch}_{time.time()}.pth')
        torch.save({'ae_state_dict': ae_net.state_dict()}, ae_net_checkpoint)

    logger.info(f'Pré-entraînement terminé. Modèle sauvegardé à : {ae_net_checkpoint}')
    return ae_net_checkpoint  # Retourne le chemin du modèle sauvegardé

# Fonction pour l'entraînement du réseau principal
def train(net, train_loader, out_dir, tb_writer, device, ae_net_checkpoint, args):
    """
    Entraîne le modèle principal (net) en utilisant l'autoencodeur pré-entraîné.
    """
    # Initialisation du logger
    logger = logging.getLogger()

    # Création d'un dictionnaire pour l'encodage
    idx_list_enc = {int(i): 1 for i in args.idx_list_enc}
    net = net.to(device)  # Déplacer le réseau sur le bon appareil (GPU/CPU)

    # Choix de l'optimiseur
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) if args.optimizer == 'adam' else SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    # Planificateur de taux d'apprentissage
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    # Initialisation des centres de sphère pour l'entraînement
    logger.info('Évaluation des centres de sphères...')
    c, keys = init_center_c(train_loader, net, idx_list_enc, device, args.end_to_end_training, args.debug)
    logger.info(f'Clés: {keys}')

    R = {k: torch.tensor(0.0, device=device) for k in keys}  # Initialisation des rayons pour chaque clé

    logger.info('Début de l\'entraînement...')
    warm_up_n_epochs = args.warm_up_n_epochs
    net.train()
    it_t = 0

    best_loss = 1e12  # Initialisation de la meilleure perte
    epochs = 1 if args.debug else args.epochs  # Nombre d'époques en fonction du mode debug
    for epoch in range(epochs):
        one_class_loss = 0.0  # Initialisation des pertes
        recon_loss = 0.0
        objective_loss = 0.0
        n_batches = 0
        d_from_c = {k: 0 for k in keys}  # Distance des points au centre des sphères
        epoch_start_time = time.time()

        # Boucle sur les lots d'entraînement
        for idx, (data, _) in enumerate(tqdm(train_loader, total=len(train_loader), desc=f"Training epoch: {epoch + 1}"), 1):
            if args.debug and idx == 2: break  # Arrêter après 2 lots si en mode debug

            n_batches += 1
            data = data.to(device)

            # Réinitialiser les gradients et calculer la perte de reconstruction
            optimizer.zero_grad()

            if args.end_to_end_training:
                x_r, _, d_lstms = net(data)
                recon_loss_ = torch.mean(torch.sum((x_r - data) ** 2, dim=tuple(range(1, x_r.dim()))))
            else:
                _, d_lstms = net(data)
                recon_loss_ = torch.tensor([0.0], device=device)

            dist, one_class_loss_ = eval_ad_loss(d_lstms, c, R, args.nu, args.boundary)
            objective_loss_ = one_class_loss_ + recon_loss_

            # Mise à jour des distances et calcul des gradients
            for k in keys:
                d_from_c[k] += torch.mean(dist[k]).item()

            objective_loss_.backward()
            optimizer.step()

            one_class_loss += one_class_loss_.item()
            recon_loss += recon_loss_.item()
            objective_loss += objective_loss_.item()

            # Affichage des informations et enregistrement dans TensorBoard
            if idx % (len(train_loader) // args.log_frequency) == 0:
                logger.info(f"Entraînement à l'époque: {epoch} [{idx}]/[{len(train_loader)}] ==> "
                            f"Perte de reconstruction: {recon_loss / n_batches:.4f}, "
                            f"Perte One Class: {one_class_loss / n_batches:.4f}, "
                            f"Perte objective: {objective_loss / n_batches:.4f}")
                tb_writer.add_scalar('train/recon_loss', recon_loss / n_batches, it_t)
                tb_writer.add_scalar('train/one_class_loss', one_class_loss / n_batches, it_t)
                tb_writer.add_scalar('train/objective_loss', objective_loss / n_batches, it_t)
                it_t += 1

        scheduler.step()  # Mise à jour du taux d'apprentissage

        # Sauvegarder le modèle à chaque époque
        time_ = time.time() if ae_net_checkpoint is None else ae_net_checkpoint
        model_checkpoint = os.path.join(out_dir, f'net_checkpoint_epoch_{epoch}_{time_}.pth')
        torch.save({'net_state_dict': net.state_dict()}, model_checkpoint)

    return model_checkpoint  # Retourner le chemin du modèle sauvegardé

@torch.no_grad()
def init_center_c(train_loader, net, idx_list_enc, device, end_to_end_training, debug, eps=0.1):
    """Initialise le centre des hypersphères (c) comme la moyenne des sorties obtenues lors d'un passage avant sur les données d'entraînement."""
    n_samples = 0  # Compteur pour le nombre total d'échantillons traités
    net.eval()  # Met le modèle en mode évaluation (désactive la mise à jour des gradients)

    # Récupère un lot d'exemples pour effectuer un premier passage avant
    data, _ = iter(train_loader).next()
    d_lstms = net(data.to(device))[-1]

    keys = []  # Liste des clés pour les centres à initialiser
    c = {}  # Dictionnaire pour stocker les centres des hypersphères
    for en, k in enumerate(list(d_lstms.keys())):
        # Si l'index de la clé fait partie de idx_list_enc, on l'ajoute à la liste des clés et on initialise le centre à zéro
        if en in idx_list_enc:
            keys.append(k)
            c[k] = torch.zeros_like(d_lstms[k][-1], device=device)

    # Parcours de l'ensemble des lots pour accumuler les valeurs du centre
    for idx, (data, _) in enumerate(tqdm(train_loader, desc='Initialisation des centres des hypersphères', total=len(train_loader), leave=False)):
        if debug and idx == 2: break  # Limite l'exécution à 2 itérations si le mode debug est activé
        # Récupère les données du lot
        n_samples += data.shape[0]  # Met à jour le compteur d'échantillons
        d_lstms = net(data.to(device))[-1]  # Effectue un passage avant sur les données
        for k in keys:
            c[k] += torch.sum(d_lstms[k], dim=0)  # Accumule les valeurs des centres

    # Calcul de la moyenne des valeurs des centres et gestion des valeurs proches de zéro
    for k in keys:
        c[k] = c[k] / n_samples  # Calcule la moyenne des centres
        # Si une valeur est proche de zéro, on l'ajuste pour éviter des valeurs triviales
        c[k][(abs(c[k]) < eps) & (c[k] < 0)] = -eps
        c[k][(abs(c[k]) < eps) & (c[k] > 0)] = eps
    
    return c, keys  # Retourne les centres calculés et les clés associées


def eval_ad_loss(d_lstms, c, R, nu, boundary):
    dist = {}  # Dictionnaire pour stocker les distances entre les points et leurs centres respectifs
    loss = 1  # Initialisation de la perte (loss)

    # Pour chaque centre dans c (les hypersphères)
    for k in c.keys():
        # Calcul de la distance au carré entre les valeurs d_lstms[k] et le centre c[k]
        dist[k] = torch.sum((d_lstms[k] - c[k].unsqueeze(0)) ** 2, dim=-1)
        
        # Si la frontière est de type 'soft', on calcule un score basé sur la distance et le rayon R[k]
        if boundary == 'soft':
            scores = dist[k] - R[k] ** 2  # Calcul du score en fonction du rayon
            # Ajoute à la perte la valeur du rayon R[k] au carré et une pénalité pour les scores supérieurs à zéro
            loss += R[k] ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            # Si la frontière n'est pas 'soft', on ajoute simplement la moyenne des distances
            loss += torch.mean(dist[k])

    return dist, loss  # Retourne les distances calculées et la perte totale
