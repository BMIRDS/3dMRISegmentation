import sys
from time import time
import pickle
from os import makedirs
from os.path import join
import argparse

import torch
from torch import nn, sigmoid
from torch.utils.data import DataLoader

from model import UNet3D
from dataset import MRIDataset
from utils import Report, transfer_weights
"""
Run inference on test set.
This script also saves voxel-wise inference results and labels on numpy arrays
for future reuse (without running on gpu).
"""

argv = sys.argv[1:]
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    prog='PROG',)
parser.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help="path to images (should have subdirs: test)")
parser.add_argument('--model_path', '-m', type=str,
                    required=True,
                    help='path to a model snapshot')
parser.add_argument('--size', type=int, nargs='+', default=[128])
parser.add_argument('--dump_name',
                    type=str,
                    default='',
                    help='name dump file to avoid overwrting files.')
args = parser.parse_args(argv)

net = UNet3D(1, 1, use_bias=True, inplanes=16)
transfer_weights(net, args.model_path)
net.cuda()
net.train(False)
torch.no_grad()

test_dir = join(args.data_dir, 'test')
batch_size = 1


def inference(target_dir):
    volume_size = args.size*3 if len(args.size) == 1 else args.size
    dataloader = DataLoader(MRIDataset(target_dir,
                                       volume_size,
                                       sampling_mode='center',
                                       deterministic=True),
                            batch_size=1,
                            num_workers=4)
    input_paths = dataloader.dataset.inputs
    label_paths = dataloader.dataset.labels

    reporter = Report()
    preds_list = list()
    labels_list = list()
    sum_hd = 0
    sum_sd = 0
    num_voxels_hd = list()
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.cuda()
        preds = sigmoid(net(inputs.detach()).detach())

        full_labels = dataloader.dataset._load_full_label(label_paths[i])
        full_preds = dataloader.dataset._project_full_label(input_paths[i],
                                                            preds.cpu())
        preds = full_preds
        labels = full_labels

        reporter.feed(preds, labels)
        temp_reporter = Report()
        temp_reporter.feed(preds, labels)
        sum_hd += temp_reporter.hard_dice()
        sum_sd += temp_reporter.soft_dice()
        num_voxels_hd.append(
            (labels.view(-1).sum().item(), temp_reporter.hard_dice()))
        del temp_reporter

        preds_list.append(preds.cpu())
        labels_list.append(labels.cpu())
        del inputs, labels, preds
    print("Micro Averaged Dice {}, {}".format(sum_hd / len(dataloader),
                                              sum_sd / len(dataloader)))
    if len(args.dump_name):
        # dump preds for visualization
        pickle.dump([input_paths, preds_list, labels_list],
                    open('preds_dump_{}.pickle'.format(args.dump_name), 'wb'))
    print(reporter)
    print(reporter.stats())
    preds = torch.stack(preds_list).view(-1).numpy()
    labels = torch.stack(labels_list).view(-1).numpy().astype(int)
    print(num_voxels_hd)
    return preds, labels


preds, labels = inference(test_dir)
