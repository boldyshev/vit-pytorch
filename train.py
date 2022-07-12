import os
import json
import argparse
import copy

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from vision_transformer import VisionTransformer
from trainer import Trainer

default_hparams = {'model': {'img_size': 32,
                             'patch_size': 4,
                             'channels': 3,
                             'num_classes': 10,
                             'embed_dim': 256,
                             'hidden_dim': 512,
                             'mlp_out': 256,
                             'num_heads': 8,
                             'num_layers': 6,
                             'drop_rate': 0.2},
                   'optimizer': {'lr': 3e-4}}

# take arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('model_name')
parser.add_argument('--max_epoch', default=128, type=int)
parser.add_argument('--batch', default=128, type=int)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--log', default=True, type=bool)
parser.add_argument('--checkpoints', default=True, type=bool)


for arg, value in default_hparams['model'].items():
    parser.add_argument(f'--{arg}', default=value, type=type(value))
parser.add_argument('--lr', default=default_hparams['optimizer']['lr'],
                    type=type(default_hparams['optimizer']['lr']))


def main():
    args = parser.parse_args()
    
    # create directories
    os.makedirs(f'models/{args.model_name}', exist_ok=True)
    checkpoints_dir = f'models/{args.model_name}/checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # assign hyperparameters
    hparams = copy.deepcopy(default_hparams)
    for arg_name in default_hparams['model'].keys():
        hparams['model'][arg_name] = vars(args)[arg_name]
    hparams['optimizer']['lr'] = args.lr

    # save hyperparameters
    path = f'models/{args.model_name}/hparams.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(hparams, f, ensure_ascii=False, indent=4)

    # get dataset
    print('Loading data...')
    data_dir = 'data'
    train_data = datasets.CIFAR10(root=data_dir, train=True, download=False)
    train_means, train_stds = train_data.data.mean(axis=(0, 1, 2)) / 255., train_data.data.std(axis=(0, 1, 2)) / 255.

    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(train_means, train_stds),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1))
                                          ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(train_means, train_stds)
                                         ])
    train_data = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
    train_data, val_data = torch.utils.data.random_split(train_data,
                                                         [45000, 5000],
                                                         generator=torch.Generator().manual_seed(42))

    batch_size = args.batch
    train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, pin_memory=True)

    test_data = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

    # create model, optimizer and learning rate scheduler
    transformer = VisionTransformer(**hparams['model'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformer.to(device)
    optimizer = torch.optim.AdamW(transformer.parameters(), **hparams['optimizer'])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)

    # train model
    vit = Trainer(transformer, optimizer, criterion, scheduler, args.model_name)
    vit.train(train_loader, val_loader, stop_epoch=args.max_epoch, resume=args.resume,
              log=args.log, checkpoints=args.checkpoints)
    vit.save_model(log=args.log)

    # test model
    test_acc = vit.test(test_loader)
    print(f'{args.model_name} test accuracy is {test_acc}')

    
if __name__ == '__main__':
    main()
