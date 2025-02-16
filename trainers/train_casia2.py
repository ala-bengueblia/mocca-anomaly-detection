import os  # Pour interagir avec le système d'exploitation (fichiers, répertoires)
import time  # Pour travailler avec le temps, horodatages et pauses
import logging  # Pour créer des journaux d'exécution
import numpy as np  # Librairie pour les calculs numériques et la manipulation de tableaux
from tqdm import tqdm  # Pour afficher des barres de progression lors des boucles longues
import torch  # Librairie pour le calcul scientifique (GPU, tenseurs)
import torch.nn.functional as F  # Fonctions d'activation et autres utilitaires de PyTorch
from torch.optim import Adam, SGD  # Optimiseurs pour entraîner les modèles (Adam, SGD)
from torch.optim.lr_scheduler import MultiStepLR  # Scheduler pour ajuster dynamiquement le taux d'apprentissage
from torch.utils.data.dataloader import DataLoader  # Pour charger les données par lots
from tensorboardX import SummaryWriter  # Pour enregistrer les métriques d'entraînement dans TensorBoard
from sklearn.metrics import roc_auc_score  # Pour calculer l'AUC-ROC, mesure de performance


def pretrain(ae_net: torch.nn.Module, train_loader: DataLoader, out_dir: str, tb_writer: SummaryWriter, device: str,
             ae_learning_rate: float, ae_weight_decay: float, ae_lr_milestones: list, ae_epochs: int) -> str:
    """
    Entraîne le réseau AutoEncoder complet pour CASIA2.

    Parameters
    ----------
    ae_net : torch.nn.Module
        Le réseau AutoEncoder à entraîner.
    train_loader : DataLoader
        Chargeur de données d'entraînement.
    out_dir : str
        Répertoire où sauvegarder les checkpoints.
    tb_writer : SummaryWriter
        Outil pour enregistrer les métriques dans TensorBoard.
    device : str
        Périphérique d'exécution (CPU ou GPU).
    ae_learning_rate : float
        Taux d'apprentissage pour l'AutoEncoder.
    ae_weight_decay : float
        Décroissance du poids (régularisation L2).
    ae_lr_milestones : list
        Liste des époques pour ajuster le taux d'apprentissage.
    ae_epochs : int
        Nombre total d'époques d'entraînement.

    Returns
    -------
    ae_net_checkpoint : str
        Chemin du checkpoint sauvegardé du modèle.
    """
    logger = logging.getLogger()
    ae_net = ae_net.train().to(device)
    optimizer = Adam(ae_net.parameters(), lr=ae_learning_rate, weight_decay=ae_weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=ae_lr_milestones, gamma=0.1)

    for epoch in range(ae_epochs):
        loss_epoch = 0.0
        n_batches = 0

        for (data, _, _) in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = ae_net(data)
            scores = torch.sum((outputs - data) ** 2, dim=tuple(range(1, outputs.dim())))
            loss = torch.mean(scores)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            n_batches += 1

        scheduler.step()
        if epoch in ae_lr_milestones:
            logger.info('LR scheduler: nouveau taux d\'apprentissage est %g' % float(scheduler.get_lr()[0]))

        logger.info(f"Pré-entraînement - Époque {epoch+1} : Perte de reconstruction = {loss_epoch/len(train_loader):.4f}")
        tb_writer.add_scalar('pretrain/recon_loss', loss_epoch/len(train_loader), epoch+1)

    logger.info('Pré-entraînement terminé.')

    ae_net_checkpoint = os.path.join(out_dir, f'ae_ckp_{time.time()}.pth')
    torch.save({'ae_state_dict': ae_net.state_dict()}, ae_net_checkpoint)
    logger.info(f'Modèle autoencodeur sauvegardé à : {ae_net_checkpoint}')

    return ae_net_checkpoint

def train(net: torch.nn.Module, train_loader: DataLoader, out_dir: str, tb_writer: SummaryWriter, device: str,
          ae_net_checkpoint: str, idx_list_enc: list, learning_rate: float, weight_decay: float,
          lr_milestones: list, epochs: int, nu: float, boundary: str, debug: bool) -> str:
    """
    Entraîne le réseau Encoder sur une tâche one-class pour CASIA2.

    Parameters
    ----------
    net : torch.nn.Module
        Réseau Encoder à entraîner.
    train_loader : DataLoader
        Chargeur de données d'entraînement.
    out_dir : str
        Répertoire de sauvegarde des checkpoints.
    tb_writer : SummaryWriter
        Outil pour enregistrer les métriques dans TensorBoard.
    device : str
        Périphérique d'exécution (CPU ou GPU).
    ae_net_checkpoint : str
        Chemin vers le checkpoint de l'autoencodeur.
    idx_list_enc : list
        Liste des indices de couches pour extraire les caractéristiques.
    learning_rate : float
        Taux d'apprentissage pour le réseau Encoder.
    weight_decay : float
        Décroissance du poids (régularisation L2).
    lr_milestones : list
        Liste des époques pour ajuster le taux d'apprentissage.
    epochs : int
        Nombre d'époques d'entraînement.
    nu : float
        Paramètre de compromis.
    boundary : str
        Type de frontière (par exemple, 'soft' ou 'hard').
    debug : bool
        Si True, active le mode débogage.

    Returns
    -------
    net_checkpoint : str
        Chemin du checkpoint sauvegardé du modèle.
    """
    logger = logging.getLogger()
    net.train().to(device)

    # Accrochage des couches du modèle pour extraire les caractéristiques
    feat_d = {}
    hooks = hook_model(idx_list_enc=idx_list_enc, model=net, dataset_name="casia2", feat_d=feat_d)

    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    logger.info('Initialisation du centre c...')
    c = init_center_c(feat_d=feat_d, train_loader=train_loader, net=net, device=device)
    logger.info('Centre c initialisé.')

    R = {k: torch.tensor(0.0, device=device) for k in c.keys()}

    logger.info('Début de l\'entraînement...')
    warm_up_n_epochs = 10

    for epoch in range(epochs):
        loss_epoch = 0.0
        n_batches = 0
        d_from_c = {}

        for (data, _, _) in train_loader:
            data = data.to(device)
            _ = net(data)
            dist, loss = eval_ad_loss(feat_d=feat_d, c=c, R=R, nu=nu, boundary=boundary)
            for k in dist.keys():
                if k not in d_from_c:
                    d_from_c[k] = 0
                d_from_c[k] += torch.mean(dist[k]).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (boundary == 'soft') and (epoch >= warm_up_n_epochs):
                for k in R.keys():
                    R[k].data = torch.tensor(
                        np.quantile(np.sqrt(dist[k].clone().data.cpu().numpy()), 1 - nu),
                        device=device
                    )

            loss_epoch += loss.item()
            n_batches += 1

        scheduler.step()
        if epoch in lr_milestones:
            logger.info('LR scheduler: nouveau taux d\'apprentissage est %g' % float(scheduler.get_lr()[0]))

        logger.info(f"Entraînement - Époque {epoch} : Perte objectif = {loss_epoch/n_batches:.4f}")
        tb_writer.add_scalar('train/objective_loss', loss_epoch/n_batches, epoch)
        for en, k in enumerate(d_from_c.keys()):
            logger.info(f"[{k}] -- Rayon: {R[k]:.4f} - Dist. du centre: {d_from_c[k]/n_batches:.4f}")
            tb_writer.add_scalar(f'train/radius_{idx_list_enc[en]}', R[k], epoch)
            tb_writer.add_scalar(f'train/distance_c_sphere_{idx_list_enc[en]}', d_from_c[k]/n_batches, epoch)

    logger.info('Entraînement terminé!')
    [h.remove() for h in hooks]

    time_ = ae_net_checkpoint.split('_')[-1].split('.p')[0]
    net_checkpoint = os.path.join(out_dir, f'net_ckp_{time_}.pth')
    if debug:
        net_checkpoint = './test_net_ckp.pth'
    torch.save({
            'net_state_dict': net.state_dict(),
            'R': R,
            'c': c
        }, net_checkpoint)
    logger.info(f'Modèle sauvegardé à : {net_checkpoint}')

    return net_checkpoint

def test(net: torch.nn.Module, test_loader: DataLoader, R: dict, c: dict, device: str, idx_list_enc: list, boundary: str) -> float:
    """
    Teste le réseau Encoder pour CASIA2.

    Parameters
    ----------
    net : torch.nn.Module
        Réseau Encoder à tester.
    test_loader : DataLoader
        Chargeur de données pour le jeu de test.
    R : dict
        Dictionnaire des rayons des hypersphères pour chaque couche.
    c : dict
        Dictionnaire des centres des hypersphères pour chaque couche.
    device : str
        Périphérique (CPU ou GPU).
    idx_list_enc : list
        Liste des indices des couches pour extraire les caractéristiques.
    boundary : str
        Type de frontière ('soft' ou autre).

    Returns
    -------
    test_auc : float
        Score AUC (Area Under the Curve) du test.
    """
    logger = logging.getLogger()
    feat_d = {}
    hooks = hook_model(idx_list_enc=idx_list_enc, model=net, dataset_name="casia2", feat_d=feat_d)

    logger.info('Démarrage du test...')
    idx_label_score = []
    net.eval().to(device)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, idx = data
            inputs = inputs.to(device)
            _ = net(inputs)
            scores = get_scores(feat_d=feat_d, c=c, R=R, device=device, boundary=boundary)
            idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                         labels.cpu().data.numpy().tolist(),
                                         scores.cpu().data.numpy().tolist()))
    [h.remove() for h in hooks]

    _, labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    test_auc = roc_auc_score(labels, scores)
    logger.info('AUC sur le jeu de test : {:.2f}%'.format(100. * test_auc))
    logger.info('Test terminé !')

    return 100. * test_auc

def hook_model(idx_list_enc: list, model: torch.nn.Module, dataset_name: str, feat_d: dict):
    """
    Crée des hooks sur les couches du modèle.

    Parameters
    ----------
    idx_list_enc : list
        Liste des indices des couches dont extraire les caractéristiques.
    model : torch.nn.Module
        Réseau Encoder à analyser.
    dataset_name : str
        Nom du jeu de données (ici 'casia2').
    feat_d : dict
        Dictionnaire pour stocker les caractéristiques extraites.

    Returns
    -------
    Liste des hooks enregistrés pour les couches spécifiées.
    """
    if dataset_name == 'mnist':
        blocks_ = [model.conv1, model.conv2, model.fc1]
    elif dataset_name == 'casia2':
        # Exemple avec une structure supposée pour CASIA2 (ajustez selon le modèle réel)
        blocks_ = [model.conv1, model.conv2, model.conv3, model.conv4, model.fc1]
    else:
        blocks_ = [model.conv1, model.conv2, model.conv3, model.fc1]

    if isinstance(idx_list_enc, list) and len(idx_list_enc) != 0:
        assert len(idx_list_enc) <= len(blocks_), f"Trop d'indices: {idx_list_enc} pour {len(blocks_)} blocs"
        blocks = [blocks_[idx] for idx in idx_list_enc]

    blocks_idx = dict(zip(blocks, map('{:02d}'.format, range(len(blocks)))))
    
    def hook_func(module, input, output):
        block_num = blocks_idx[module]
        extracted = output
        if extracted.ndimension() > 2:
            extracted = F.avg_pool2d(extracted, extracted.shape[-2:])
        feat_d[block_num] = extracted.squeeze()

    return [b.register_forward_hook(hook_func) for b in blocks_idx]

@torch.no_grad()
def init_center_c(feat_d: dict, train_loader: DataLoader, net: torch.nn.Module, device: str, eps: float = 0.1) -> dict:
    """
    Initialise le centre c des hypersphères comme la moyenne d'une passe avant initiale.

    Parameters
    ----------
    feat_d : dict
        Caractéristiques extraites.
    train_loader : DataLoader
        Chargeur de données d'entraînement.
    net : torch.nn.Module
        Réseau Encoder.
    device : str
        Périphérique (CPU ou GPU).
    eps : float, optional
        Valeur minimale pour éviter un centre trop proche de zéro.

    Returns
    -------
    c : dict
        Centres calculés pour chaque couche.
    """
    n_samples = 0
    net.eval()
    for idx, (data, _, _) in enumerate(tqdm(train_loader, desc='Init centres hypersphériques', total=len(train_loader), leave=False)):
        data = data.to(device)
        outputs = net(data)
        n_samples += outputs.shape[0]
        if idx == 0:
            c = {k: torch.zeros_like(feat_d[k][-1], device=device) for k in feat_d.keys()}
        for k in feat_d.keys():
            c[k] += torch.sum(feat_d[k], dim=0)
    for k in c.keys():
        c[k] = c[k] / n_samples
        c[k][(abs(c[k]) < eps) & (c[k] < 0)] = -eps
        c[k][(abs(c[k]) < eps) & (c[k] > 0)] = eps
    return c

def eval_ad_loss(feat_d: dict, c: dict, R: dict, nu: float, boundary: str) -> [dict, torch.Tensor]:
    """
    Évalue la perte d'entraînement pour la détection d'anomalies.

    Parameters
    ----------
    feat_d : dict
        Caractéristiques extraites.
    c : dict
        Centres des hypersphères.
    R : dict
        Rayons des hypersphères.
    nu : float
        Paramètre de compromis.
    boundary : str
        Type de frontière ('soft' ou autre).

    Returns
    -------
    dist : dict
        Distances moyennes par couche.
    loss : torch.Tensor
        Valeur de la perte.
    """
    dist = {}
    loss = 1
    for k in feat_d.keys():
        dist[k] = torch.sum((feat_d[k] - c[k].unsqueeze(0)) ** 2, dim=1)
        if boundary == 'soft':
            scores = dist[k] - R[k] ** 2
            loss += R[k] ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss += torch.mean(dist[k])
    return dist, loss

def get_scores(feat_d: dict, c: dict, R: dict, device: str, boundary: str) -> float:
    """
    Calcule le score d'anomalie.

    Parameters
    ----------
    feat_d : dict
        Caractéristiques extraites.
    c : dict
        Centres des hypersphères.
    R : dict
        Rayons des hypersphères.
    device : str
        Périphérique (CPU ou GPU).
    boundary : str
        Type de frontière ('soft' ou autre).

    Returns
    -------
    scores : float
        Score d'anomalie moyen.
    """
    dist, _ = eval_ad_loss(feat_d, c, R, 1, boundary)
    shape = dist[list(dist.keys())[0]].shape[0]
    scores = torch.zeros((shape,), device=device)
    for k in dist.keys():
        if boundary == 'soft':
            scores += dist[k] - R[k] ** 2
        else:
            scores += dist[k]
    return scores / len(list(dist.keys()))