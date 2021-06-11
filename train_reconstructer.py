from __future__ import print_function
from architectures import DENOISERS_ARCHITECTURES, get_architecture, IMAGENET_CLASSIFIERS, CLASSIFIERS_ARCHITECTURES
from datasets import get_dataset, DATASETS

import argparse
import numpy as np
import os
import csv
import math
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torch.utils.data as Data
import torch.backends.cudnn as cudnn

import models
from utils import progress_bar


# Checkpoint related
START_EPOCH = 0

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        
        
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    cs_model.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # Forward pass
        noise = torch.randn_like(inputs) * args.noise_sd
        rec_in = model((normalize(inputs) + noise.cuda()))
        
        outputs = cs_model(rec_in)

        loss1 = criterion(outputs, targets)*(1e-1)
        loss2 = (rec_in - inputs).abs().sum() / ((rec_in - inputs)!=0).sum()
        loss = loss1 + loss2

        train_loss += loss.item()
        _, pred_idx = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += pred_idx.eq(targets.data).cpu().sum().float()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(train_loader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/batch_idx, 100.*correct/total



def test(epoch):
    model.eval()
    cs_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # Forward pass
        noise = torch.randn_like(inputs) * args.noise_sd
        rec_in = model((normalize(inputs) + noise.cuda()))
        
        outputs = cs_model(rec_in)

        loss1 = criterion(outputs, targets)*(1e-1)
        loss2 = (rec_in - inputs).abs().sum() / ((rec_in - inputs)!=0).sum()
        loss = loss1 + loss2

        test_loss += loss.item()
        _, pred_idx = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += pred_idx.eq(targets.data).cpu().sum().float()

        progress_bar(batch_idx, len(test_loader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return test_loss/batch_idx, 100.*correct/total


def save_checkpoint(acc, epoch):
    print('=====> Saving checkpoint...')
    state = {
        'model': model,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    torch.save(state, args.save_dir + args.name + '.ckpt')


# Decrease the learning rate at 5 and 10 and 15 epoch
def adjust_lr(optimizer, epoch):
    lr = args.lr

    if args.dataset == 'imagenet':
        if epoch >= 5:
            lr /= 10
        if epoch >= 10:
            lr /= 10
        if epoch >= 15:
            lr /= 10
    elif args.dataset == 'cifar10':
        if epoch >= 15:
            lr /= 10
        if epoch >= 30:
            lr /= 10
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    ## Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=DATASETS)
    # Directory
    parser.add_argument('--data-dir', default='data/', help='data path')
    parser.add_argument('--save-dir', type=str, help='save dir for reconstructer model')
    parser.add_argument('--pretrained', type=str, help='pretrained model storage path')
    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default=0.1)')
    parser.add_argument('--noise-sd', type=float, default=0.12, help='sd for noise')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=20, help='total epochs')
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_ckpt_steps', type=int, default=10, help='save checkpoint steps (default: 10)')

    # Utility parameters
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', type=str, default='UNet', help='choose reconstructer model type (default: ResNet18)')
    parser.add_argument('--model_cs', type=str, default='ResNet18', help='choose classification model type (default: ResNet18)')

    parser.add_argument('--name', type=str, help='name of the run')

    args = parser.parse_args()

    # Data
    print('=====> Preparing data...')

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        batch_size = args.batch_size 

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               shuffle=True, 
                                               batch_size=batch_size,
                                               num_workers=8, 
                                               pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              shuffle=False, 
                                              batch_size=batch_size,
                                              num_workers=8, 
                                              pin_memory=pin_memory)

    # Models
    if args.resume:
        print('=====> Resuming from checkpoint...')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.save_dir + args.name + '.ckpt')

        model = checkpoint['model']
        acc = checkpoint['acc']
        START_EPOCH = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        print('=====> Building model...')
        model = models.__dict__[args.model](3,3)

    if args.pretrained:
        print('=====> Loading Pretrained Classifier...')
        checkpoint = torch.load(args.pretrained)
        cs_model = get_architecture(checkpoint['arch'], args.dataset)
        cs_model.load_state_dict(checkpoint['state_dict'], strict=False)


    model = model.to(device)

    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        
    logname = ('results/log_' + args.name + '.csv')

    if (torch.cuda.device_count() > 1) and (args.dataset=='imagenet'):
        print("=====> Use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
        cudnn.benchmark = True
        

    # # Entropy-Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])

    for epoch in range(START_EPOCH, args.epoch):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        adjust_lr(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

        
        save_checkpoint(test_acc, epoch)
