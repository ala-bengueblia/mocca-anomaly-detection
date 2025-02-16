# main_micc_f220.py
import os
import sys
import glob
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import makedirs
from os.path import exists
from prettytable import PrettyTable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets.micc_data import MICCDataManager
from models.micc_model import MICC_F220_AutoEncoder, MICC_F220_Encoder
from trainers.trainer_micc import pretrain, train, test
from utils import (set_seeds, get_out_dir, eval_spheres_centers,load_micc_model_from_checkpoint, purge_ae_params)

def test_models(test_loader: DataLoader, net_checkpoint: str, tables: tuple, 
               out_df: pd.DataFrame, input_shape: tuple, device: str, 
               debug: bool) -> pd.DataFrame:
    """Évalue les modèles sur le dataset MICC-F220"""
    logger = logging.getLogger()
    
    if not os.path.exists(net_checkpoint):
        logger.error(f"Checkpoint introuvable : {net_checkpoint}")
        return out_df

    try:
        # Extraction des paramètres du modèle
        path_parts = net_checkpoint.split('/')
        code_length = int(path_parts[-3].split('_')[-1])
        boundary = path_parts[-4].split('-')[-1]
        normal_class = path_parts[-5]

        # Chargement du modèle
        net = load_micc_model_from_checkpoint(
            input_shape=input_shape,
            code_length=code_length,
            net_checkpoint=net_checkpoint
        )
        net.load_state_dict(torch.load(net_checkpoint)['net_state_dict'])

        # Exécution du test
        test_auc, test_bacc = test(
            net=net,
            test_loader=test_loader,
            device=device,
            debug=debug
        )

        # Mise à jour des résultats
        table = tables[0] if boundary == 'soft' else tables[1]
        table.add_row([
            normal_class,
            code_length,
            boundary,
            f"{test_auc:.4f}",
            f"{test_bacc:.4f}"
        ])

        out_df = out_df.append({
            'Classe': normal_class,
            'Code latent': code_length,
            'Frontière': boundary,
            'AUC': test_auc,
            'Accuracy équilibrée': test_bacc
        }, ignore_index=True)

    except Exception as e:
        logger.error(f"Erreur lors du test {net_checkpoint}: {str(e)}")

    return out_df

def main(args):
    # Configuration initiale
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler('micc_training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # Gestion des données
    data_manager = MICCDataManager(
        data_root=args.data_path,
        normal_class=args.normal_class,
        image_size=(220, 220),
        test_only=args.test
    )
    
    # Chargement des données
    train_loader, test_loader = data_manager.get_loaders(
        batch_size=args.batch_size,
        num_workers=args.n_workers
    )

    # Configuration modèle
    input_shape = (3, 220, 220)  # Format MICC-F220

    # Phase de pré-entraînement
    if args.pretrain:
        logger.info("Démarrage du pré-entraînement...")
        ae_model = MICC_F220_AutoEncoder(code_length=args.code_length)
        ae_model.to(device)
        
        pretrain(
            model=ae_model,
            train_loader=train_loader,
            epochs=args.ae_epochs,
            lr=args.ae_lr,
            device=device,
            output_dir=args.output_dir
        )

    # Phase d'entraînement
    if args.train:
        logger.info("Démarrage de l'entraînement...")
        encoder = MICC_F220_Encoder(code_length=args.code_length)
        encoder.to(device)
        
        if args.pretrained_path:
            purge_ae_params(encoder, args.pretrained_path)

        train(
            model=encoder,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            nu=args.nu,
            device=device,
            output_dir=args.output_dir
        )

    # Phase de test
    if args.test:
        logger.info("Démarrage des tests...")
        results_table = PrettyTable()
        results_table.field_names = ["Classe", "Code latent", "Frontière", "AUC", "Balanced Accuracy"]
        
        results_df = pd.DataFrame()

        checkpoints = glob.glob(os.path.join(args.checkpoint_dir, "*.pth"))
        for ckpt in tqdm(checkpoints, desc="Évaluation des modèles"):
            results_df = test_models(
                test_loader=test_loader,
                net_checkpoint=ckpt,
                tables=(results_table, results_table),  # Adaptation pour MICC
                out_df=results_df,
                input_shape=input_shape,
                device=device,
                debug=args.debug
            )

        logger.info("\nRésultats finaux :")
        logger.info("\n" + str(results_table))
        results_df.to_csv(os.path.join(args.output_dir, "micc_results.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Détection d'anomalies MICC-F220")

    # Paramètres dataset
    parser.add_argument("--data-path", default="./MICC-F220", help="Chemin vers le dataset")
    parser.add_argument("--normal-class", default="authentique", choices=["authentique", "tampon1", "tampon2"],help="Classe normale à considérer")

    # Configuration modèle
    parser.add_argument("--code-length", type=int, default=256, help="Dimension de l'espace latent")
    parser.add_argument("--pretrained-path", help="Chemin vers un modèle pré-entraîné")

    # Hyperparamètres
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--ae-lr", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--nu", type=float, default=0.05, help="Paramètre de régularisation")

    # Configuration exécution
    parser.add_argument("--output-dir", default="./results_micc")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--debug", action="store_true")

    # Opérations
    parser.add_argument("--pretrain", action="store_true", help="Exécuter le pré-entraînement")
    parser.add_argument("--train", action="store_true", help="Exécuter l'entraînement")
    parser.add_argument("--test", action="store_true", help="Exécuter les tests")

    args = parser.parse_args()

    # Création des répertoires
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    main(args)