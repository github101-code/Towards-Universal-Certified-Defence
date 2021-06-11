from architectures import get_architecture, IMAGENET_CLASSIFIERS
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
from time import time

import argparse
import datetime
import os
import torch

import numpy as np
import sys
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", default = 'cifar10', choices=DATASETS, help="which dataset")
parser.add_argument('--ckpt-dir', default='', help='checkpoint path of the reconstructer')
parser.add_argument("--base_classifier", type=str, default = '', help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=int, default = 0.12, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=256, help="batch size")
parser.add_argument("--skip", type=int, default=100, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--name', type=str, help='name of saved checkpoint of reconstucter')


parser.add_argument('--azure_datastore_path', type=str, default='',
                    help='Path to imagenet on azure')
parser.add_argument('--philly_imagenet_path', type=str, default='',
                    help='Path to imagenet on philly')
args = parser.parse_args("".split())

if args.azure_datastore_path:
    os.environ['IMAGENET_DIR_AZURE'] = os.path.join(args.azure_datastore_path, 'datasets/imagenet_zipped')
elif args.philly_imagenet_path:
    os.environ['IMAGENET_DIR_PHILLY'] = os.path.join(args.philly_imagenet_path, './')
else:
    os.environ['IMAGENET_DIR_PHILLY'] = "/hdfs/public/imagenet/2012/"
    
    
    
class defence(nn.Module):
    
    def __init__(self, model):
        super(defence, self).__init__()
        self.model = model

    def forward(self, input):
        x = normalize(input)
        n = (torch.randn(x.shape) * args.sigma).cuda()
        return self.model(x+n)

    
if __name__ == "__main__":
    
    dataset = get_dataset(args.dataset, args.split)

    # load the base classifier
    if args.base_classifier in IMAGENET_CLASSIFIERS:
        assert args.dataset == 'imagenet'
        # loading pretrained imagenet architectures
        base_classifier = get_architecture(args.base_classifier ,args.dataset, pytorch_pretrained=True)
    else:
        checkpoint = torch.load(args.base_classifier)
        base_classifier = get_architecture(checkpoint['arch'], args.dataset)
        base_classifier.load_state_dict(checkpoint['state_dict'])

    if args.name != '':
        checkpoint = torch.load(args.ckpt_dir + args.name + '.ckpt')
        model = checkpoint['model']
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
        base_classifier = torch.nn.Sequential(model, base_classifier)

    base_classifier = base_classifier.eval()
    defence_model = defence(base_classifier)

    # create the smooothed classifier g
    smoothed_classifier = Smooth(defence_model, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    if not os.path.exists(args.outfile.split('sigma')[0]):
        os.makedirs(args.outfile.split('sigma')[0])

    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)
    f.close()

    # iterate through the dataset

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        x = x.cuda()
        before_time = time()
        # certify the prediction of g around x
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        f = open(args.outfile, 'a')
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), flush=True)
        f.close()
