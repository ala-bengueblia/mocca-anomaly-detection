import os  # Pour interagir avec le système de fichiers (créer des dossiers, etc.)
import random  # Pour générer des nombres aléatoires
import logging  # Pour enregistrer les messages de débogage et d'information
import numpy as np  # Pour les calculs numériques
from tqdm import tqdm  # Pour afficher des barres de progression
from PIL import Image  # Pour manipuler des images
from os.path import join  # Pour manipuler des chemins de fichiers
import torch  # La bibliothèque PyTorch pour le deep learning
import torchvision.transforms as T  # Pour appliquer des transformations aux images
from torch.utils.data import DataLoader  # Pour charger des données en batches
from torchvision.datasets import ImageFolder  # Pour charger des datasets d'images
from models.mvtec_model import MVTec_Encoder  # Modèle d'encodeur pour MVTecr


def get_out_dir(args, pretrain: bool, aelr: float, dset_name: str="cifar10", training_strategy: str=None) -> [str, str]:
    """Creates training output dir

    Parameters
    ----------

    args : 
        Arguments
    pretrain : bool
        True if pretrain the model
    aelr : float
        Full AutoEncoder learning rate
    dset_name : str
        Dataset name
    training_strategy : str
        ................................................................
    
    Returns
    -------
    out_dir : str
        Path to output folder
    tmp : str
        String containing infos about the current experiment setup

    """
    if dset_name == "ShanghaiTech":
        if pretrain:
            # Crée un répertoire pour le prétraining
            tmp = (f"pretrain-mn_{dset_name}-cl_{args.code_length}-lr_{args.ae_learning_rate}")
            out_dir = os.path.join(args.output_path, dset_name, 'pretrain', tmp)
        else:
            # Crée un répertoire pour l'entraînement
            tmp = (
                f"train-mn_{dset_name}-cl_{args.code_length}-bs_{args.batch_size}-nu_{args.nu}-lr_{args.learning_rate}-"
                f"bd_{args.boundary}-sl_{args.use_selectors}-ile_{'.'.join(map(str, args.idx_list_enc))}-lstm_{args.load_lstm}-"
                f"bidir_{args.bidirectional}-hs_{args.hidden_size}-nl_{args.num_layers}-dp_{args.dropout}"
            )
            out_dir = os.path.join(args.output_path, dset_name, 'train', tmp)
            
            # Crée un répertoire pour l'entraînement end-to-end
            if args.end_to_end_training:
                out_dir = os.path.join(args.output_path, dset_name, 'train_end_to_end', tmp)
    else:
        if pretrain: 
            # Crée un répertoire pour le prétraining sur d'autres datasets  
            tmp = (f"pretrain-mn_{dset_name}-nc_{args.normal_class}-cl_{args.code_length}-lr_{args.ae_learning_rate}-awd_{args.ae_weight_decay}")
            out_dir = os.path.join(args.output_path, dset_name, str(args.normal_class), 'pretrain', tmp)
        
        else:
            # Crée un répertoire pour l'entraînement sur d'autres datasets
            tmp = (
                f"train-mn_{dset_name}-nc_{args.normal_class}-cl_{args.code_length}-bs_{args.batch_size}-nu_{args.nu}-lr_{args.learning_rate}-"
                f"wd_{args.weight_decay}-bd_{args.boundary}-alr_{aelr}-sl_{args.use_selectors}-ep_{args.epochs}-ile_{'.'.join(map(str, args.idx_list_enc))}"
            )
            out_dir = os.path.join(args.output_path, dset_name, str(args.normal_class), 'train', tmp)
    # Crée le répertoire s'il n'existe pas
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir, tmp # Retourne le chemin du répertoire et une chaîne descriptive

def set_seeds(seed: int) -> None:
    """Set all seeds.
    
    Parameters
    ----------
    seed : int
        Seed

    """
    # Set the seed only if the user specified it
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def purge_params(encoder_net, ae_net_cehckpoint: str) -> None:
    """Load Encoder preatrained weights from the full AutoEncoder.
    After the pretraining phase, we don't need the full AutoEncoder parameters, we only need the Encoder
    
    Parameters
    ----------
    encoder_net :
        The Encoder network
    ae_net_cehckpoint : str
        Path to full AutoEncoder checkpoint
    
    """
    # Load the full AutoEncoder checkpoint dict
    ae_net_dict = torch.load(ae_net_cehckpoint, map_location=lambda storage, loc: storage)['ae_state_dict']# Charge le checkpoint
        
    # Load encoder weight from autoencoder
    net_dict = encoder_net.state_dict()# Récupère les poids de l'encodeur
    
    # Filter out decoder network keys
    st_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}# Filtre les clés du décodeur
    
    # Overwrite values in the existing state_dict
    net_dict.update(st_dict)# Met à jour les poids de l'encodeur
    
    # Load the new state_dict
    encoder_net.load_state_dict(net_dict)# Charge les nouveaux poids
        
def load_mvtec_model_from_checkpoint(input_shape: tuple, code_length: int, idx_list_enc: list, use_selectors: bool, net_cehckpoint: str, purge_ae_params: bool = False) -> torch.nn.Module:
    """Load AutoEncoder checkpoint. 
    
    Parameters
    ----------
    input_shape : tuple
        Input data shape
    code_length : int
        Latent code size
    idx_list_enc : list
        List of indexes of layers from which extract features
    use_selectors : bool
        True if the model has to use Selector modules
    net_cehckpoint : str
        Path to model checkpoint
    purge_ae_params : bool 
        True if the checkpoint is relative to an AutoEncoder

    Returns
    -------
    encoder_net : torch.nn.Module
        The Encoder network

    """
    logger = logging.getLogger()

    # Initialise l'encodeur
    encoder_net = MVTec_Encoder(
                            input_shape=input_shape,
                            code_length=code_length,
                            idx_list_enc=idx_list_enc,
                            use_selectors=use_selectors
                        )
    
    if purge_ae_params:
        # Charge les poids de l'encodeur à partir de l'auto-encodeur prétraîné
        # Load Encoder parameters from pretrianed full AutoEncoder
        logger.info(f"Loading encoder from: {net_cehckpoint}")
        purge_params(encoder_net=encoder_net, ae_net_cehckpoint=net_cehckpoint)
    else:
        # Charge les poids directement à partir du checkpoint
        st_dict = torch.load(net_cehckpoint)
        encoder_net.load_state_dict(st_dict['net_state_dict'])
        logger.info(f"Loaded model from: {net_cehckpoint}")
    
    return encoder_net # Retourne l'encodeur chargé

def extract_arguments_from_checkpoint(net_checkpoint: str):
    """Takes file path of the checkpoint and parse the checkpoint name to extract training parameters and
    architectural specifications of the model. 
    
    Parameters
    ----------
    net_checkpoint : file path of the checkpoint (str) 

    Returns
    -------
    code_length = latent code size (int)
    batch_size = batch_size (int)
    boundary = soft or hard boundary (str)
    use_selectors = if selectors used it is true, otherwise false (bool)
    idx_list_enc = indexes of the exploited layers (list of integers)
    load_lstm = boolean to show whether lstm used (bool)
    hidden_size = hidden size of the lstm (int)
    num_layers = number of layers of the lstm (int)
    dropout = dropout probability (float)
    bidirectional = is lstm bi-directional or not (bool)
    dataset_name = name of the dataset (str)
    train_type = is it end-to-end, train, or pretrain (str)
    """

    code_length = int(net_checkpoint.split(os.sep)[-2].split('-')[2].split('_')[-1])# Longueur du code latent
    batch_size = int(net_checkpoint.split(os.sep)[-2].split('-')[3].split('_')[-1])# Taille du batch
    boundary = net_checkpoint.split(os.sep)[-2].split('-')[6].split('_')[-1]# Type de frontière
    use_selectors = net_checkpoint.split(os.sep)[-2].split('-')[7].split('_')[-1] == "True" # Utilisation des selecteurs
    idx_list_enc = [int(i) for i in net_checkpoint.split(os.sep)[-2].split('-')[8].split('_')[-1].split('.')]# Liste des indices des couches
    load_lstm = net_checkpoint.split(os.sep)[-2].split('-')[9].split('_')[-1] == "True" # Utilisation des LSTMs
    hidden_size = int(net_checkpoint.split(os.sep)[-2].split('-')[11].split('_')[-1])# Taille cachée des LSTMs
    num_layers = int(net_checkpoint.split(os.sep)[-2].split('-')[12].split('_')[-1])# Nombre de couches des LSTMs
    dropout = float(net_checkpoint.split(os.sep)[-2].split('-')[13].split('_')[-1])# Probabilité de dropout
    bidirectional = net_checkpoint.split(os.sep)[-2].split('-')[10].split('_')[-1] == "True"  # LSTMs bidirectionnels
    dataset_name = net_checkpoint.split(os.sep)[-4] # Nom du dataset
    train_type = net_checkpoint.split(os.sep)[-3]  # Type d'entraînement
    return code_length, batch_size, boundary, use_selectors, idx_list_enc, load_lstm, hidden_size, num_layers, dropout, bidirectional, dataset_name, train_type

def eval_spheres_centers(train_loader: DataLoader, encoder_net: torch.nn.Module, ae_net_cehckpoint: str, use_selectors: bool, device:str, debug: bool) -> dict:
    """Eval the centers of the hyperspheres at each chosen layer.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader for trainin data
    encoder_net : torch.nn.Module
        Encoder network 
    ae_net_cehckpoint : str
        Checkpoint of the full AutoEncoder 
    use_selectors : bool
        True if we want to use selector models
    device : str
        Device on which run the computations
    debug : bool
        Activate debug mode
    
    Returns
    -------
    dict : dictionary
        Dictionary with k='layer name'; v='features vector representing hypersphere center' 
    
    """
    logger = logging.getLogger()
    
    # Nom du fichier des centres
    centers_files = ae_net_cehckpoint[:-4]+f'_w_centers_{use_selectors}.pth'

    # If centers are found, then load and return
    if os.path.exists(centers_files):
    
        logger.info("Found hyperspheres centers")
        ae_net_ckp = torch.load(centers_files, map_location=lambda storage, loc: storage)

        centers = {k: v.to(device) for k, v in ae_net_ckp['centers'].items()}
    else:
        # Sinon, évalue les centres
        logger.info("Hyperspheres centers not found... evaluating...")
        centers_ = init_center_c(train_loader=train_loader, encoder_net=encoder_net, device=device, debug=debug)
        
        logger.info("Hyperspheres centers evaluated!!!")
        new_ckp = ae_net_cehckpoint.split('.pth')[0]+f'_w_centers_{use_selectors}.pth'
        
        logger.info(f"New AE dict saved at: {new_ckp}!!!")
        centers = {k: v for k, v in centers_.items()}
        
        torch.save({
                'ae_state_dict': torch.load(ae_net_cehckpoint)['ae_state_dict'],
                'centers': centers
                }, new_ckp)

    return centers  #Retourne les centres des hypersphères

@torch.no_grad()
def init_center_c(train_loader: DataLoader, encoder_net: torch.nn.Module, device: str, debug: bool, eps: float=0.1) -> dict:
    """Initialize hypersphere center as the mean from an initial forward pass on the data.
    
    Parameters
    ----------
    train_loader : 
    encoder_net : 
    debug : 
    eps: 

    Returns
    -------
    dictionary : dict
        Dictionary with k='layer name'; v='center featrues'

    """
    n_samples = 0

    encoder_net.eval().to(device) # Met l'encodeur en mode évaluation

    for idx, (data, _) in enumerate(tqdm(train_loader, desc='Init hyperspheres centeres', total=len(train_loader), leave=False)):
        if debug and idx == 5: break  # Mode debug : limite à 5 batches
    
        data = data.to(device)
        n_samples += data.shape[0]

        zipped = encoder_net(data)# Passe les données à travers l'encodeur
        
        if idx == 0:
            # Initialise les centres
            c = {item[0]: torch.zeros_like(item[1][-1], device=device) for item in zipped}
    
        for item in zipped:
            # Accumule les caractéristiques
            c[item[0]] += torch.sum(item[1], dim=0)
    
    for k in c.keys():
        c[k] = c[k] / n_samples # Calcule la moyenne
    
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        # Évite les valeurs trop proches de zéro
        c[k][(abs(c[k]) < eps) & (c[k] < 0)] = -eps
        c[k][(abs(c[k]) < eps) & (c[k] > 0)] = eps

    return c  # Retourne les centres des hypersphères
    
