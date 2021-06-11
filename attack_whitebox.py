from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from architectures import DENOISERS_ARCHITECTURES, get_architecture, IMAGENET_CLASSIFIERS, CLASSIFIERS_ARCHITECTURES
from datasets import get_dataset, DATASETS

import argparse
import numpy as np
import os
import sys

import tensorflow as tf
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data

from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
# Directory
parser.add_argument('--dataset', type=str, default='cifar10', choices=DATASETS)
parser.add_argument('--ckpt-dir', type=str, help='checkpoint path')
parser.add_argument('--pretrained', type=str, help='pretrained_model path')
parser.add_argument('--noise-sd', type=float, default=0.25, help='sd for noise')
parser.add_argument('--name', type=str, help='name of saved checkpoints')

# Attack parameters
parser.add_argument('--source', type=str, default='baseline',
                    help='source model of transfer-based black-box attacks')
parser.add_argument('--attack-type', type=str, default='pgd',
                    choices=['fgsm', 'pgd', 'cw'])
parser.add_argument('--epsilon', type=float, default=8, help='The upper bound change of L-inf norm on input pixels')
parser.add_argument('--iter', type=int, default=100, help='The number of iterations for iterative attacks')
parser.add_argument('--cw-conf', type=int, default=100, help='The confidence of adversarial examples for CW attack')

args = parser.parse_args()

config = {
    'epsilon': args.epsilon / 255.,
    'num_steps': args.iter,
    'step_size': 2 / 255.,
    'random_start': True,
}


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalization param
mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])




class defence(nn.Module):

    def __init__(self, model):
        super(defence, self).__init__()
        self.model = model

    def forward(self, input):
        x = normalize(input)
        n = torch.randn(x.shape).cuda()*args.noise_sd
        return self.model(x + n)


def transfer_attack():
        
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        if args.attack_type == 'fgsm':
            adversarial = fast_gradient_method(model_fn = defence_model, x = inputs, eps=config['epsilon'], \
                          norm = np.inf, clip_min=0, clip_max = 1, y = targets)        
        elif args.attack_type == 'pgd':
            adversarial = projected_gradient_descent(model_fn = defence_model, x = inputs, eps=config['epsilon'], \
                          eps_iter=config['step_size'], nb_iter = config['num_steps'], norm = np.inf, clip_min=0,\
                          clip_max = 1, y = targets, rand_init = config['random_start'])
        else:
            adversarial = carlini_wagner_l2(model_fn = defence_model, x = inputs, n_classes=10, lr=config['step_size'], \
                                            confidence = args.cw_conf, max_iterations=config['num_steps'],\
                                            clip_min=0, clip_max = 1, y = targets)

        outputs = defence_model(adversarial)
        _, pred_idx = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += pred_idx.eq(targets.data).cpu().sum().float()

        sys.stdout.write("\rTransfer-based white-box %s attack... Acc: %.3f%% (%d/%d)" %
                         (args.attack_type, 100. * correct / (total), correct, total))
        sys.stdout.flush()

    print('Accuracy under transfer-based %s attack: %.3f%%' % (args.attack_type, 100. * correct / total))


def test_generalization():
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = defence_model(inputs)

            _, pred_idx = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += pred_idx.eq(targets.data).cpu().sum().float()

            sys.stdout.write("\rGeneralization... Acc: %.3f%% (%d/%d)"
                             % (100. * correct / total, correct, total))
            sys.stdout.flush()

    return 100. * correct / total


if __name__ == '__main__':

    print('=====> Preparing data...')

    dataset = get_dataset(args.dataset, 'test')

    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=100,
                                            shuffle=False,
                                            num_workers=8)

    print('=====> Loading trained model from checkpoint...')
    checkpoint = torch.load(args.ckpt_dir + args.name + '.ckpt')

    model = checkpoint['model']

    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

    if args.pretrained:
        print('=====> Loading Pretrained Classifier...')
        checkpoint = torch.load(args.pretrained)
        cs_model = get_architecture(checkpoint['arch'], 'cifar10')
        cs_model.load_state_dict(checkpoint['state_dict'], strict=False)


    model = torch.nn.Sequential(model, cs_model).to(device)
    defence_model = defence(model)

    defence_model.eval()

    # Generalization
    print('=====> Generalization of trained model... Acc: %.3f%%' % test_generalization())
    # Adversarial robustness
    print('=====> White-box BPDA on trained model... Acc: %.3f%%' % transfer_attack())
