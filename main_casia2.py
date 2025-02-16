# main_casia2.py
import os
import sys
import random
import logging
import argparse
import numpy as np
import torch
from tensorboardX import SummaryWriter
from datasets.casia_data import CASIADataManager
from trainers.train_casia2 import pretrain, train, test
from utils import set_seeds, get_out_dir, purge_ae_params
from models.casia2_model import CASIA2_Autoencoder, CASIA2_Encoder

def main(args):
    if len(args.idx_list_enc) == 0 and args.train:
        args.idx_list_enc = [3]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler('./casia_training.log'),
            logging.StreamHandler()
        ])

    logger = logging.getLogger()
    
    if args.train or args.pretrain:
        logger.info(
            "\nCASIA2 Training Configuration:"
            f"\n\nMode:"
            f"\n\tPretrain: {args.pretrain}"
            f"\n\tTrain: {args.train}"
            f"\n\tTest: {args.test}"
            f"\n\nModel Parameters:"
            f"\n\tBoundary type: {args.boundary}"
            f"\n\tNormal class: {args.normal_class}"
            f"\n\tCode length: {args.code_length}"
            f"\n\tEncoder layers: {args.idx_list_enc}"
            f"\n\nOptimization:"
            f"\n\tBatch size: {args.batch_size}"
            f"\n\tAE Learning rate: {args.ae_learning_rate}"
            f"\n\tAE Milestones: {args.ae_lr_milestones}"
            f"\n\tAE Weight decay: {args.ae_weight_decay}"
            f"\n\tTraining epochs: {args.epochs}"
            f"\n\tNu parameter: {args.nu}"
        )

    if args.test and not args.model_ckp:
        logger.error("Testing requires a valid model checkpoint!")
        sys.exit(1)

    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CASIA2 Data Loading
    data_manager = CASIADataManager(
        data_root=args.data_path,
        normal_class=args.normal_class,
        image_size=(256, 256),
        test_only=args.test
    )
    
    train_loader, test_loader = data_manager.get_loaders(
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        pin_memory=device=="cuda"
    )

    # Pretrain Autoencoder
    ae_checkpoint = None
    if args.pretrain:
        out_dir = os.path.join(args.output_path, f'casia2_class_{args.normal_class}')
        tb_writer = SummaryWriter(log_dir=os.path.join(out_dir, 'pretrain_logs'))
        
        ae_net = CASIA2_Autoencoder(args.code_length).to(device)
        
        logger.info('Starting Autoencoder Pretraining...')
        ae_checkpoint = pretrain(
            ae_net=ae_net,
            train_loader=train_loader,
            out_dir=out_dir,
            tb_writer=tb_writer,
            device=device,
            ae_learning_rate=args.ae_learning_rate,
            ae_weight_decay=args.ae_weight_decay,
            ae_lr_milestones=args.ae_lr_milestones,
            ae_epochs=args.ae_epochs
        )
        logger.info(f'Autoencoder trained. Saved to: {ae_checkpoint}')
        tb_writer.close()

    # Train Encoder
    model_checkpoint = None
    if args.train:
        if not ae_checkpoint and not args.model_ckp:
            logger.error("Encoder training requires pretrained autoencoder!")
            sys.exit(1)
        
        out_dir = os.path.join(args.output_path, f'casia2_class_{args.normal_class}')
        tb_writer = SummaryWriter(log_dir=os.path.join(out_dir, 'train_logs'))
        
        encoder = CASIA2_Encoder(args.code_length).to(device)
        purge_ae_params(encoder, ae_checkpoint or args.model_ckp)

        logger.info('Starting Encoder Training...')
        model_checkpoint = train(
            net=encoder,
            train_loader=train_loader,
            out_dir=out_dir,
            tb_writer=tb_writer,
            device=device,
            ae_net_checkpoint=ae_checkpoint,
            idx_list_enc=args.idx_list_enc,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_milestones=args.lr_milestones,
            epochs=args.epochs,
            nu=args.nu,
            boundary=args.boundary,
            debug=args.debug
        )
        logger.info(f'Encoder trained. Saved to: {model_checkpoint}')
        tb_writer.close()

    # Test Model
    if args.test:
        encoder = CASIA2_Encoder(args.code_length).to(device)
        checkpoint = torch.load(args.model_ckp)
        encoder.load_state_dict(checkpoint['net_state_dict'])
        
        logger.info(f'Loaded model from: {args.model_ckp}')
        logger.info(f'Testing Configuration:'
                    f"\n\tNormal class: {args.normal_class}"
                    f"\n\tBoundary: {checkpoint['boundary']}"
                    f"\n\tEncoder layers: {checkpoint['layers']}")

        test_results = test(
            net=encoder,
            test_loader=test_loader,
            R=checkpoint['R'],
            c=checkpoint['c'],
            device=device,
            idx_list_enc=checkpoint['layers'],
            boundary=checkpoint['boundary']
        )
        
        logger.info(f'Test Results - AUC: {test_results:.2%}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CASIA2 Anomaly Detection')
    
    # Dataset
    parser.add_argument('--data-path', default='./casia2', help='Path to CASIA2 dataset')
    parser.add_argument('--normal-class', type=int, default=0, choices=range(10), help='Normal class index (0-9)')
    
    # Model
    parser.add_argument('--code-length', type=int, default=128, help='Latent space dimension')
    parser.add_argument('--model-ckp', help='Path to trained model checkpoint')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--ae-learning-rate', type=float, default=1e-4)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--ae-weight-decay', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--ae-epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--nu', type=float, default=0.1)
    
    # Operations
    parser.add_argument('--pretrain', action='store_true', help='Pretrain autoencoder')
    parser.add_argument('--train', action='store_true', help='Train encoder')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    main(args)