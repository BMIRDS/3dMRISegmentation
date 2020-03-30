from datetime import datetime
from os.path import join
import argparse
import sys

from torch import sigmoid, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

from dataset import MRIDataset
from loss import DiceLoss
from model import UNet3D
from utils import (get_weight_vector, Report,
                   transfer_weights)

argv = sys.argv[1:]
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    prog='PROG',)
parser.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help="path to images (should have subdirs: train, val)")
parser.add_argument('--save_dir',
                    type=str,
                    required=True,
                    help="path to save snapshots of trained models")
parser.add_argument('--learning_rate', '-r',
                    type=float,
                    default='1e-3',
                    help='initial learning rate for adam')
parser.add_argument('--weight_decay',
                    type=float,
                    default=1e-4,
                    help='weight decay, use for model regularization')
parser.add_argument('--volume_size', '-v',
                    type=int,
                    nargs='+',
                    default=[128],
                    help='input volume size (x,y,z) to network.'
                    "When only one value is given, it expanded to three dim.")
parser.add_argument('--weight', '-w',
                    type=int,
                    default=1,
                    help='relative weight of positive samples for bce loss')
parser.add_argument('--epochs', type=int, default=300,
                    help="the total number of training epochs")
parser.add_argument('--restart', type=int, default=50,
                    help='restart learning rate every <restart> epochs')
parser.add_argument('--resume_model',
                    type=str,
                    default=None,
                    help='path to load previously saved model')
args = parser.parse_args(argv)
print(args)

is_cuda = torch.cuda.is_available()

net = UNet3D(1, 1, use_bias=True, inplanes=16)
if args.resume_model is not None:
    transfer_weights(net, args.resume_model)
bce_crit = nn.BCELoss()
dice_crit = DiceLoss()
last_bce_loss = 0
last_dice_loss = 0


def criterion(pred, labels, weights=[0.1, 0.9]):
    _bce_loss = bce_crit(pred, labels)
    _dice_loss = dice_crit(pred, labels)
    global last_bce_loss, last_dice_loss
    last_bce_loss = _bce_loss.item()
    last_dice_loss = _dice_loss.item()
    return weights[0] * _bce_loss + weights[1] * _dice_loss


size = args.volume_size * 3 if len(args.volume_size) == 1 else args.volume_size
assert len(size) == 3
relative_weight = args.weight
save_dir = args.save_dir

train_dir = join(args.data_dir, 'train')
train_loader = DataLoader(MRIDataset(train_dir,
                                     size,
                                     sampling_mode='random',
                                     deterministic=True),
                          shuffle=True,
                          batch_size=1,
                          pin_memory=True)

val_dir = join(args.data_dir, 'val')
val_loader = DataLoader(MRIDataset(val_dir,
                                   size,
                                   sampling_mode='center_val',
                                   deterministic=True),
                        batch_size=1,
                        pin_memory=True)

optimizer = optim.Adam(net.parameters(),
                       lr=args.learning_rate,
                       weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer,
                              T_max=args.restart * len(train_loader))

if is_cuda:
    net = net.cuda()
    bce_crit = bce_crit.cuda()
    dice_crit = dice_crit.cuda()


def train(train_loader, epoch):
    net.train(True)
    reporter = Report()
    epoch_bce_loss = 0
    epoch_dice_loss = 0
    epoch_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        if is_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = sigmoid(net(inputs))
        reporter.feed(outputs, labels)
        bce_crit.weight = get_weight_vector(labels, relative_weight, is_cuda)
        loss = criterion(outputs, labels)
        epoch_bce_loss += last_bce_loss
        epoch_dice_loss += last_dice_loss
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        del inputs, labels, outputs, loss
    avg_bce_loss = epoch_bce_loss / float(len(train_loader))
    avg_dice_loss = epoch_dice_loss / float(len(train_loader))
    avg_loss = epoch_loss / float(len(train_loader))
    avg_acc = reporter.accuracy()
    print("\n[Train] Epoch({}) Avg BCE Loss: {:.4f} Avg Dice Loss: {:.4f} \
        Avg Loss: {:.4f}".format(epoch, avg_bce_loss, avg_dice_loss, avg_loss))
    print(reporter)
    print(reporter.stats())
    return avg_loss, avg_acc


def validate(val_loader, epoch):
    net.train(False)
    reporter = Report()
    epoch_bce_loss = 0
    epoch_dice_loss = 0
    epoch_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            preds = sigmoid(net(inputs.detach()).detach())
            reporter.feed(preds, labels)
            bce_crit.weight = get_weight_vector(labels, relative_weight,
                                                is_cuda)
            loss = criterion(preds, labels)
            epoch_bce_loss += last_bce_loss
            epoch_dice_loss += last_dice_loss
            epoch_loss += loss.item()
            del inputs, labels, preds, loss
    avg_bce_loss = epoch_bce_loss / float(len(val_loader))
    avg_dice_loss = epoch_dice_loss / float(len(val_loader))
    avg_loss = epoch_loss / float(len(val_loader))
    avg_acc = reporter.accuracy()
    print("[Valid] Epoch({}) Avg BCE Loss: {:.4f} Avg Dice Loss: {:.4f} \
        Avg Loss: {:.4f}".format(epoch, avg_bce_loss, avg_dice_loss, avg_loss))
    print(reporter)
    print(reporter.stats())
    return avg_loss, avg_acc


if __name__ == "__main__":
    best_performance = float('Inf')
    n_epochs = args.epochs
    for epoch in range(n_epochs):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        train_loss, train_acc = train(train_loader, epoch)
        valid_loss, valid_acc = validate(val_loader, epoch)

        if valid_loss < best_performance:
            best_performance = valid_loss
            torch.save(
                net,
                join(save_dir, 'net-epoch-{:03}.pth'.format(epoch)))
            print("model saved")
        if epoch > args.restart and epoch % args.restart == 0:
            scheduler.last_epoch = -1
            print("lr restart")
        sys.stdout.flush()
