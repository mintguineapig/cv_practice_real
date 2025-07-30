import numpy as np
import os
import random
import wandb

import torch
import argparse
import timm
import logging
from tqdm import tqdm

from train import fit
from models import *
from datasets import create_dataset, create_dataloader
from log import setup_default_logging

_logger = logging.getLogger('train')


def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def get_model(model_name, num_classes, img_size=32, pretrained=False):
    print(f">>> Creating model: {model_name} with pretrained={pretrained}, img_size={img_size}")
    
    try:
        # Vision Transformer models need img_size parameter
        if 'vit' in model_name.lower() or 'deit' in model_name.lower():
            model = timm.create_model(model_name, 
                                     pretrained=pretrained, 
                                     num_classes=num_classes, 
                                     img_size=img_size)
        else:
            # CNN models don't need img_size parameter
            model = timm.create_model(model_name, 
                                     pretrained=pretrained, 
                                     num_classes=num_classes)
        return model
    except Exception as e:
        print(f"Failed to create model '{model_name}': {e}")
        print("Use timm.list_models() to check available models.")
        raise ValueError(f"Unsupported model: {model_name}")


def run(args):
    savedir = os.path.join(args.savedir, args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # CIFAR10/100: 32x32, Tiny ImageNet: 64x64
    img_size = 32 if args.dataname in ['CIFAR10','CIFAR100'] else 64
    _logger.info(f'Image size: {img_size}')
    _logger.info(f'Augmentation: {args.aug_name}')

    model = get_model(args.model_name, args.num_classes, img_size=img_size, pretrained=args.pretrained)
    model.to(device)
    
    _logger.info('Model: {}'.format(args.model_name))
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    trainset, testset = create_dataset(datadir=args.datadir, dataname=args.dataname, aug_name=args.aug_name)
    
    trainloader = create_dataloader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    testloader = create_dataloader(dataset=testset, batch_size=256, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.opt_name](model.parameters(), lr=args.lr)

    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    wandb.init(name=args.exp_name, project='DSBA-study', config=args)

    fit(model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        epochs       = args.epochs, 
        savedir      = savedir,
        log_interval = args.log_interval,
        device       = device)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Classification for Computer Vision")
    
    parser.add_argument('--model-name', type=str, default='resnet18',
                       help='timm model name (e.g., resnet18, efficientnet_b0, vit_base_patch16_224)')
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained weights')
    
    parser.add_argument('--exp-name', type=str, help='experiment name')
    parser.add_argument('--datadir', type=str, default='/datasets', help='data directory')
    parser.add_argument('--savedir', type=str, default='./saved_model', help='saved model directory')

    parser.add_argument('--dataname', type=str, default='CIFAR100', 
                       choices=['CIFAR10','CIFAR100', 'tiny-imagenet-200'], help='target dataset')
    parser.add_argument('--num-classes', type=int, default=100, help='number of classes')

    parser.add_argument('--opt-name', type=str, choices=['SGD','Adam'], help='optimizer name')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

    parser.add_argument('--use_scheduler', action='store_true', help='use scheduler')

    parser.add_argument('--aug-name', type=str, 
                       choices=['default','weak','strong', 'cutmix', 'mixup', 'cifar'], 
                       help='augmentation type')

    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval')

    parser.add_argument('--seed', type=int, default=1222, help='random seed')

    args = parser.parse_args()
    run(args)
