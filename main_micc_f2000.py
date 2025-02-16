import os  # Pour interagir avec le système de fichiers (créer des dossiers, etc.)
import sys  # Pour gérer les arguments de ligne de commande et quitter le programme si nécessaire
import random  # Pour générer des nombres aléatoires
import logging  # Pour enregistrer les messages de débogage et d'information
import argparse  # Pour parser les arguments passés en ligne de commande
import numpy as np  # Pour les calculs numériques
import matplotlib.pyplot as plt  # Pour visualiser des graphiques
from torchvision.utils import make_grid  # Pour créer une grille d'images (utile pour visualiser des batchs)
import torch  # La bibliothèque PyTorch pour le deep learning
import torch.nn as nn  # Pour définir des modèles de réseaux de neurones
from tensorboardX import SummaryWriter  # Pour visualiser les métriques d'entraînement avec TensorBoard
from models.shanghaitech_model import ShanghaiTech, ShanghaiTechEncoder, ShanghaiTechDecoder  # Modèles spécifiques pour le dataset ShanghaiTech
from datasets.data_manager import DataManager  # Classe pour gérer le chargement des données
from datasets.shanghaitech_test import VideoAnomalyDetectionResultHelper  # Classe pour aider à tester les résultats de détection d'anomalies vidéo
from trainers.trainer_shanghaitech import pretrain, train  # Fonctions pour prétraîner et entraîner le modèle
from utils import set_seeds, get_out_dir, eval_spheres_centers, load_mvtec_model_from_checkpoint, extract_arguments_from_checkpoint  # Utilitaires pour définir les seeds, obtenir le répertoire de sortie, évaluer les centres des hypersphères, charger un modèle et extraire des arguments d'un checkpoint


def main(args):
    # Set seed
    # Définit les seeds pour la reproductibilité
    set_seeds(args.seed)

    # Get the device
    # Vérifie si un GPU est disponible et définit le device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Désactive le logging si spécifié
    if args.disable_logging:
        logging.disable(level=logging.INFO)

    
    ## Init logger & print training/warm-up summary
    logging.basicConfig(
        level=logging.INFO, # Niveau de logging (INFO)
        format="%(asctime)s | %(message)s",# Format des messages de log
        handlers=[
            logging.FileHandler('./training.log'),# Enregistre les logs dans un fichier
            logging.StreamHandler()# Affiche les logs dans la console
        ])
    logger = logging.getLogger()# Récupère le logger

    # Si l'entraînement, le prétraining ou l'entraînement end-to-end est activé
    if args.train or args.pretrain or args.end_to_end_training:

        # If the list of layers from which extract the features is empty, then use the last one (after the sigmoid)
        if len(args.idx_list_enc) == 0: args.idx_list_enc = [6]

        # Affiche les paramètres d'entraînement dans le fichier de log
        logger.info(
                "Start run with params:\n"
                f"\n\t\t\t\tEnd to end training : {args.end_to_end_training}"
                f"\n\t\t\t\tPretrain model      : {args.pretrain}"
                f"\n\t\t\t\tTrain model         : {args.train}"
                f"\n\t\t\t\tTest model          : {args.test}"
                f"\n\t\t\t\tBatch size          : {args.batch_size}\n"
                f"\n\t\t\t\tAutoEncoder Pretraining"
                f"\n\t\t\t\tPretrain epochs     : {args.ae_epochs}"
                f"\n\t\t\t\tAE-Learning rate    : {args.ae_learning_rate}"
                f"\n\t\t\t\tAE-milestones       : {args.ae_lr_milestones}"
                f"\n\t\t\t\tAE-Weight decay     : {args.ae_weight_decay}\n"
                f"\n\t\t\t\tEncoder Training"
                f"\n\t\t\t\tClip length         : {args.clip_length}"
                f"\n\t\t\t\tBoundary            : {args.boundary}"
                f"\n\t\t\t\tTrain epochs        : {args.epochs}"
                f"\n\t\t\t\tLearning rate       : {args.learning_rate}"
                f"\n\t\t\t\tMilestones          : {args.lr_milestones}"
                f"\n\t\t\t\tUse selectors       : {args.use_selectors}"
                f"\n\t\t\t\tWeight decay        : {args.weight_decay}"
                f"\n\t\t\t\tCode length         : {args.code_length}"
                f"\n\t\t\t\tNu                  : {args.nu}"
                f"\n\t\t\t\tEncoder list        : {args.idx_list_enc}\n"
                f"\n\t\t\t\tLSTMs"
                f"\n\t\t\t\tLoad LSTMs          : {args.load_lstm}"
                f"\n\t\t\t\tBidirectional       : {args.bidirectional}"
                f"\n\t\t\t\tHidden size         : {args.hidden_size}"
                f"\n\t\t\t\tNumber of layers    : {args.num_layers}"
                f"\n\t\t\t\tDropout prob        : {args.dropout}\n"
            )
    else:
        # Si aucun checkpoint n'est fourni pour le test, quitte le programme
        if args.model_ckp is None:
            logger.info("CANNOT TEST MODEL WITHOUT A VALID CHECKPOINT")
            sys.exit(0)
        
    # Définit le device (GPU ou CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init DataHolder class
    # Initialise la classe DataManager pour gérer les données
    data_holder = DataManager(
                        dataset_name='ShanghaiTech', # Nom du dataset
                        data_path=args.data_path, # Chemin des données
                        normal_class=None, # Classe normale (non utilisé ici)
                        only_test=args.test  # Mode test uniquement
                    ).get_data_holder()

    # Load data
    # Charge les données en utilisant DataLoader
    train_loader, _ = data_holder.get_loaders(
        batch_size=args.batch_size, # Taille du batch
        shuffle_train=True, # Mélange les données
        pin_memory=device=="cuda", # Utilise la mémoire paginée si GPU
        num_workers=args.n_workers # Nombre de workers pour le chargement des données
        )
    # Print data infos  
    # Affiche les informations sur les données
    only_test = args.test and not args.train and not args.pretrain
    logger.info("Dataset info:")
    logger.info(
            "\n"
            f"\n\t\t\t\tBatch size    : {args.batch_size}"    
        )
    if not only_test:
        logger.info(
                f"TRAIN:"
                f"\n\t\t\t\tNumber of clips  : {len(train_loader.dataset)}"
                f"\n\t\t\t\tNumber of batches : {len(train_loader.dataset)//args.batch_size}"
            )


    ########################################################################################
    ####### Train the AUTOENCODER on the RECONSTRUCTION task and then train only the #######
    ########################## ENCODER on the ONE CLASS OBJECTIVE ##########################
    ########################################################################################
    ae_net_checkpoint = None
    if args.pretrain and not args.end_to_end_training:
         # Obtient le répertoire de sortie pour le prétraining
        out_dir, tmp = get_out_dir(args, pretrain=True, aelr=None, dset_name='ShanghaiTech')

        # Initialise TensorBoard pour le prétraining
        tb_writer = SummaryWriter(os.path.join(args.output_path, "ShanghaiTech", 'tb_runs_pretrain', tmp))
        # Init AutoEncoder
        ae_net = ShanghaiTech(data_holder.shape, args.code_length,use_selectors=args.use_selectors) 
        ### PRETRAIN
        ae_net_checkpoint = pretrain(ae_net, train_loader, out_dir, tb_writer, device, args)
        tb_writer.close()

    net_checkpoint = None
    
    if args.train and not args.end_to_end_training:
        if ae_net_checkpoint is None:
            if args.model_ckp is None:
                logger.info("CANNOT TRAIN MODEL WITHOUT A VALID CHECKPOINT")
                sys.exit(0)
            ae_net_checkpoint = args.model_ckp

        # Extrait le taux d'apprentissage du checkpoint
        aelr = float(ae_net_cehckpoint.split('/')[-2].split('-')[4].split('_')[-1])
        
        # Obtient le répertoire de sortie pour l'entraînement
        out_dir, tmp = get_out_dir(args, pretrain=False, aelr=aelr, dset_name='ShanghaiTech')
       
        # Initialise TensorBoard pour l'entraînement
        tb_writer = SummaryWriter(os.path.join(args.output_path, "ShanghaiTech", 'tb_runs_train', tmp))

        # Init Encoder
        net = ShanghaiTechEncoder(data_holder.shape, args.code_length, args.load_lstm, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, args.use_selectors)

        # Load encoder weight from autoencoder
        # Charge les poids de l'encodeur à partir de l'auto-encodeur prétraîné
        net_dict = net.state_dict()
        logger.info(f"Loading encoder from: {ae_net_checkpoint}")
        ae_net_dict = torch.load(ae_net_checkpoint, map_location=lambda storage, loc: storage)['ae_state_dict']

        # Filter out decoder network keys
        st_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(st_dict)
        # Load the new state_dict
        net.load_state_dict(net_dict)

        ### TRAIN
        net_checkpoint = train(net, train_loader, out_dir, tb_writer, device, ae_net_checkpoint, args)
        tb_writer.close()

    ########################################################################################
    ########################################################################################
    
    ########################################################################################
    ################### Train the AUTOENCODER on the combined objective: ###################
    ############################## RECONSTRUCTION + ONE CLASS ##############################
    ########################################################################################
    if args.end_to_end_training:
        
        # Obtient le répertoire de sortie pour l'entraînement end-to-end
        out_dir, tmp = get_out_dir(args, pretrain=False, aelr=int(args.learning_rate), dset_name='ShanghaiTech')
    
        # Initialise TensorBoard pour l'entraînement end-to-end
        tb_writer = SummaryWriter(os.path.join(args.output_path, "ShanghaiTech", 'tb_runs_train_end_to_end', tmp))
        # Init AutoEncoder
        ae_net = ShanghaiTech(data_holder.shape, args.code_length, args.load_lstm, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, args.use_selectors) 
        ### End to end TRAIN
        net_checkpoint = train(ae_net, train_loader, out_dir, tb_writer, device, None, args)
        tb_writer.close()
    ########################################################################################
    ########################################################################################
    
    ########################################################################################
    ###################################### Model test ######################################
    ########################################################################################
    if args.test:   
        if net_checkpoint is None:
            net_checkpoint = args.model_ckp
        
        # Extrait les arguments du checkpoint
        code_length, batch_size, boundary, use_selectors, idx_list_enc, \
        load_lstm, hidden_size, num_layers, dropout, bidirectional, \
        dataset_name, train_type = extract_arguments_from_checkpoint(net_checkpoint)

        # Init dataset
        dataset = data_holder.get_test_data()
        if train_type == "train_end_to_end":
            # Init Autoencoder
            net = ShanghaiTech(data_holder.shape, args.code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional, use_selectors) 
        else:
            # Init Encoder ONLY
            net = ShanghaiTechEncoder(dataset.shape, code_length, load_lstm, hidden_size, num_layers, dropout, bidirectional, use_selectors) 
        st_dict = torch.load(net_checkpoint)

        # Charge les poids du modèle
        net.load_state_dict(st_dict['net_state_dict'])
        logger.info(f"Loaded model from: {net_checkpoint}")
        logger.info(
                f"Start test with params:"
                f"\n\t\t\t\tDataset        : {dataset_name}"
                f"\n\t\t\t\tCode length    : {code_length}"
                f"\n\t\t\t\tEnc layer list : {idx_list_enc}"
                f"\n\t\t\t\tBoundary       : {boundary}"
                f"\n\t\t\t\tUse Selectors  : {use_selectors}"
                f"\n\t\t\t\tBatch size     : {batch_size}"
                f"\n\t\t\t\tN workers      : {args.n_workers}"
                f"\n\t\t\t\tLoad LSTMs     : {load_lstm}"
                f"\n\t\t\t\tHidden size    : {hidden_size}"
                f"\n\t\t\t\tNum layers     : {num_layers}"
                f"\n\t\t\t\tBidirectional  : {bidirectional}"
                f"\n\t\t\t\tDropout prob   : {dropout}"
            )

        # Initialize test helper for processing each video seperately
        # It prints the result to the loaded checkpoint directory
        helper = VideoAnomalyDetectionResultHelper(
                                                dataset=dataset,
                                                model=net,
                                                c=st_dict['c'], 
                                                R=st_dict['R'], 
                                                boundary=boundary,
                                                device=device,
                                                end_to_end_training= True if train_type == "train_end_to_end" else False,
                                                debug=args.debug,
                                                output_file=os.path.join("".join(net_checkpoint.split(os.sep)[:-1]),"shanghaitech_test_results.txt")
                                            )
        ### TEST
        helper.test_video_anomaly_detection()
        print("Test finished")
    ########################################################################################
    ########################################################################################
    
if __name__ == '__main__':

     # Parse les arguments en ligne de commande
    parser = argparse.ArgumentParser('AD')
    
    ## General config
    parser.add_argument('-s', '--seed', type=int, default=-1, help='Random seed (default: -1)')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers for data loading. 0 means that the data will be loaded in the main process. (default=8)')
    parser.add_argument('--output_path', default='./output')
    parser.add_argument('-lf', '--log-frequency', type=int, default=5, help='Log frequency (default: 5)')
    parser.add_argument('-dl', '--disable-logging', action="store_true", help='Disabel logging (default: False)')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug mode (default: False)')
    ## Model config
    parser.add_argument('-zl', '--code-length', default=2048, type=int, help='Code length (default: 2048)')
    parser.add_argument('-ck', '--model-ckp', help='Model checkpoint')
    ## Optimizer config
    parser.add_argument('-opt', '--optimizer', choices=('adam', 'sgd'), default='adam', help='Optimizer (default: adam)')
    parser.add_argument('-alr', '--ae-learning-rate', type=float, default=1.e-4, help='Warm up learning rate (default: 1.e-4)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1.e-4, help='Learning rate (default: 1.e-4)')
    parser.add_argument('-awd', '--ae-weight-decay', type=float, default=0.5e-6, help='Warm up learning rate (default: 1.e-4)')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.5e-6, help='Learning rate (default: 1.e-4)')
    parser.add_argument('-aml', '--ae-lr-milestones', type=int, nargs='+', default=[], help='Pretrain milestone')
    parser.add_argument('-ml', '--lr-milestones', type=int, nargs='+', default=[], help='Training milestone')
    ## Data
    parser.add_argument('-dp', '--data-path', default='./ShanghaiTech', help='Dataset main path')
    parser.add_argument('-cl', '--clip-length', type=int, default=16, help='Clip length (default: 16)')
    ## Training config
    # LSTMs
    parser.add_argument('-ll', '--load-lstm', action="store_true", help='Load LSTMs (default: False)')
    parser.add_argument('-bdl', '--bidirectional', action="store_true", help='Bidirectional LSTMs (default: False)')
    parser.add_argument('-hs', '--hidden-size', type=int, default=100, help='Hidden size (default: 100)')
    parser.add_argument('-nl', '--num-layers', type=int, default=1, help='Number of LSTMs layers (default: 1)')
    parser.add_argument('-drp', '--dropout', type=float, default=0.0, help='Dropout probability (default: 0.0)')
    # Autoencoder
    parser.add_argument('-ee', '--end-to-end-training', action="store_true", help='End-to-End training of the autoencoder (default: False)')
    parser.add_argument('-we', '--warm_up_n_epochs', type=int, default=5, help='Warm up epochs (default: 5)')
    parser.add_argument('-use','--use-selectors', action="store_true", help='Use features selector (default: False)')
    parser.add_argument('-ba', '--batch-accumulation', type=int, default=-1, help='Batch accumulation (default: -1, i.e., None)')
    parser.add_argument('-ptr', '--pretrain', action="store_true", help='Pretrain model (default: False)')
    parser.add_argument('-tr', '--train', action="store_true", help='Train model (default: False)')
    parser.add_argument('-tt', '--test', action="store_true", help='Test model (default: False)')
    parser.add_argument('-tbc', '--train-best-conf', action="store_true", help='Train best configurations (default: False)')
    parser.add_argument('-bs', '--batch-size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('-bd', '--boundary', choices=("hard", "soft"), default="soft", help='Boundary (default: soft)')
    parser.add_argument('-ile', '--idx-list-enc', type=int, nargs='+', default=[], help='List of indices of model encoder')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Training epochs (default: 1)')
    parser.add_argument('-ae', '--ae-epochs', type=int, default=1, help='Warmp up epochs (default: 1)')
    parser.add_argument('-nu', '--nu', type=float, default=0.1)

    args = parser.parse_args()
    main(args)