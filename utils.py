from collections import defaultdict
from os.path import join
from random import randint
from scipy import ndimage
from statistics import median
import numpy
import os
import shutil
import sys

from torch import nn
import torch
import nibabel as nib


def transfer_weights(target_model, saved_model):
    """
    target_model: a model instance whose weight params are to be overwritten
    saved_model: a model whose weight params will be transfered to target.
        saved_model can be a string(path to a snapshot), an instance of model
        or a state dict of a model
    """
    target_dict = target_model.state_dict()
    if isinstance(saved_model, str):
        source_dict = torch.load(saved_model)
    else:
        source_dict = saved_model
    if not isinstance(source_dict, dict):
        source_dict = source_dict.state_dict()
    source_dict = {k: v for k, v in source_dict.items() if
                   k in target_model.state_dict() and source_dict[k].size() == target_model.state_dict()[k].size()}
    target_dict.update(source_dict)
    target_model.load_state_dict(target_dict)


def generate_ex_list(directory):
    """
    Generate list of MRI objects
    """
    inputs = []
    labels = []
    for dirpath, dirs, files in os.walk(directory):
        label_list = list()
        for file in files:
            if not file.startswith('.') and file.endswith('.nii.gz'):
                if ("Lesion" in file):
                    label_list.append(join(dirpath, file))
                elif ("mask" not in file):
                    inputs.append(join(dirpath, file))
        if label_list:
            labels.append(label_list)

    return inputs, labels


def gen_mask(lesion_files):
    """
    Given a list of lesion files, generate a mask
    that incorporates data from all of them
    """
    first_lesion = nib.load(lesion_files[0]).get_data()
    if len(lesion_files) == 1:
        return first_lesion
    lesion_data = numpy.zeros((first_lesion.shape[0], first_lesion.shape[1], first_lesion.shape[2]))
    for file in lesion_files:
        l_file = correct_dims(nib.load(file).get_data())
        lesion_data = numpy.maximum(l_file, lesion_data)
    return lesion_data


def correct_dims(img):
    """
    Fix the dimension of the image, if necessary
    """
    if len(img.shape) > 3:
        img = img.reshape(img.shape[0], img.shape[1], img.shape[2])
    return img


def get_weight_vector(labels, weight, is_cuda):
    """ Generates the weight vector for BCE loss
    You can only control positive weight, and negative weight is
    default to 1.
    So if ratio of positive and negative samples are 1:3,
    then give weight 3, and this functio returns 3 for positive and
    1 for negative samples.
    """
    if is_cuda:
        labels = labels.cpu()
    labels = labels.data.numpy()
    labels = labels * (weight-1) + 1
    weight_label = torch.from_numpy(labels).type(torch.FloatTensor)
    if is_cuda:
        weight_label = weight_label.cuda()
    return weight_label


def resize_img(input_img, label_img, size):
    """
    size: int or list of int
        when it's a list, it should include x, y, z values
    Resize image to (size x size x size)
    """
    if isinstance(size, int):
        size = [size]*3
    assert len(size) == 3
    ax1 = float(size[0]) / input_img.shape[0]
    ax2 = float(size[1]) / input_img.shape[1]
    ax3 = float(size[2]) / input_img.shape[2]
    ex = ndimage.zoom(input_img, (ax1, ax2, ax3))
    label = ndimage.zoom(label_img, (ax1, ax2, ax3))
    return ex, label


def center_crop(input_img, label_img, size):
    """
    Crop center section from image
    size: int or list of int
        when it's a list, it should include x, y, z values
    Use for testing.
    """
    if isinstance(size, int):
        size = [size]*3
    assert len(size) == 3
    coords = [0]*3
    for i in range(3):
        coords[i] = int((input_img.shape[i]-size[i])//2)
    x, y, z = coords
    ex = input_img[x:x+size[0], y:y+size[1], z:z+size[2]]
    label = label_img[x:x+size[0], y:y+size[1], z:z+size[2]]
    return ex, label


def find_and_crop_lesions(input_img, label_img, size, deterministic=False):
    """
    Find and crop image based on center of lesions
    size: int or list of int
        when it's a list, it should include x, y, z values
    Use for validation.
    """
    if isinstance(size, int):
        size = [size]*3
    assert len(size) == 3
    nonzeros = label_img.nonzero()
    d = [0]*3
    if not deterministic:
        for i in range(3):
            d[i] = randint(-size[i]//4, size[i]//4)

    coords = [0]*3
    for i in range(3):
        coords[i] = max(min(int(median(nonzeros[i])) - (size[i] // 2) + d[i], input_img.shape[i] - size[i] - 1), 0)
    x, y, z = coords
    ex = input_img[x:x+size[0], y:y+size[1], z:z+size[2]]
    label = label_img[x:x+size[0], y:y+size[1], z:z+size[2]]
    return ex, label


def random_crop(input_img, label_img, size, remove_background=False):
    """
    Crop random section from image
    size: int or list of int
        when it's a list, it should include x, y, z values
    remove_background: boolean
        use this option when input contains larger background or crop size is very small
    Use for training
    """
    if isinstance(size, int):
        size = [size]*3
    assert len(size) == 3
    non_zero_percentage = 0
    while non_zero_percentage < 0.7:
        """draw x,y,z coords
        """
        coords = [0]*3
        for i in range(3):
            coords[i] = numpy.random.choice(input_img.shape[i] - size[i])
        x, y, z = coords
        ex = input_img[x:x+size[0], y:y+size[1], z:z+size[2]]
        non_zero_percentage = numpy.count_nonzero(ex) / float(size[0]*size[1]*size[2])
        if not remove_background:
            break
        if non_zero_percentage < 0.7:
            del ex

    label = label_img[x:x+size[0], y:y+size[1], z:z+size[2]]
    return ex, label


class Report:
    EPS = sys.float_info.epsilon
    TP_KEY = 0
    TN_KEY = 1
    FP_KEY = 2
    FN_KEY = 3

    def __init__(self, threshold=0.5, smooth=sys.float_info.epsilon, apply_square=False, need_feedback=False):
        """
        apply_square: use squared elements in the denominator of soft Dice
        need_feedback: returns a tensor storing KEYS(0 to 3) for each output element
        """
        self.pos = 0
        self.neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.true_pos = 0
        self.true_neg = 0
        self.soft_I = 0
        self.soft_U = 0
        self.hard_I = 0
        self.hard_U = 0
        self.smooth = smooth
        self.apply_square = apply_square  # this variable: mainly for testing
        self.need_feedback = need_feedback
        self.threshold = threshold
        self.pathdic = defaultdict(list)

    def feed(self, pred, label, paths=None):
        """ pred size: batch x dim1 x dim2 x...
            label size: batch x dim1 x dim2 x...
            First dim should be a batch size
        """
        self.soft_I += (pred * label).sum().item()
        power_coeff = 2 if self.apply_square else 1
        if power_coeff == 1:
            self.soft_U += (pred.sum() + label.sum()).item()
        else:
            self.soft_U += (pred.pow(power_coeff).sum() + label.pow(power_coeff).sum()).item()
        pred = pred.view(-1)
        label = label.view(-1)
        pred = (pred > self.threshold).squeeze()
        not_pred = (pred == 0).squeeze()
        label = label.byte().squeeze()
        not_label = (label == 0).squeeze()
        self.pos += label.sum().item()
        self.neg += not_label.sum().item()
        pxl = pred * label
        self.hard_I += (pxl).sum().item()
        self.hard_U += (pred.sum() + label.sum()).item()
        pxnl = pred * not_label
        fp = (pxnl).sum().item()
        self.false_pos += fp
        npxl = not_pred * label
        fn = (npxl).sum().item()
        self.false_neg += fn
        tp = (pxl).sum().item()
        self.true_pos += tp
        npxnl = not_pred * not_label
        tn = (npxnl).sum().item()
        self.true_neg += tn

        feedback = None
        if self.need_feedback:
            feedback = pxl*self.TP_KEY +\
                npxnl*self.TN_KEY +\
                pxnl*self.FP_KEY +\
                npxl*self.FN_KEY
            if paths is not None:
                # Variable -> list of int
                feedback_int = [int(feedback.data[i]) for i in range(feedback.numel())]
                for i in range(len(feedback_int)):
                    if feedback_int[i] == self.TP_KEY:
                        self.pathdic["TP"].append(paths[i])
                    elif feedback_int[i] == self.TN_KEY:
                        self.pathdic["TN"].append(paths[i])
                    elif feedback_int[i] == self.FP_KEY:
                        self.pathdic["FP"].append(paths[i])
                    elif feedback_int[i] == self.FN_KEY:
                        self.pathdic["FN"].append(paths[i])
        return feedback

    def stats(self):
        text = ("Total Positives: {}".format(self.pos),
                "Total Negatives: {}".format(self.neg),
                "Total TruePos: {}".format(self.true_pos),
                "Total TrueNeg: {}".format(self.true_neg),
                "Total FalsePos: {}".format(self.false_pos),
                "Total FalseNeg: {}".format(self.false_neg))
        return "\n".join(text)

    def accuracy(self):
        return (self.true_pos+self.true_neg) / max((self.pos+self.neg), self.EPS)

    def hard_dice(self):
        numer = 2 * self.hard_I + self.smooth
        denom = self.hard_U + self.smooth
        return numer / denom

    def soft_dice(self):
        numer = 2 * self.soft_I + self.smooth
        denom = self.soft_U + self.smooth
        return numer / denom

    def __summarize(self):
        self.ACC = self.accuracy()
        self.HD = self.hard_dice()
        self.SD = self.soft_dice()

        self.P_TPR = self.true_pos / max(self.pos, self.EPS)
        self.P_PPV = self.true_pos / max((self.true_pos + self.false_pos), self.EPS)
        self.P_F1 = 2*self.true_pos / max((2*self.true_pos + self.false_pos + self.false_neg), self.EPS)

        self.N_TPR = self.true_neg / max(self.neg, self.EPS)
        self.N_PPV = self.true_neg / max((self.true_neg + self.false_neg), self.EPS)
        self.N_F1 = 2*self.true_neg / max((2*self.true_neg + self.false_neg + self.false_pos), self.EPS)

    def __str__(self):
        self.__summarize()
        summary = ("Accuracy: {:.4f}".format(self.ACC),
                   "Hard Dice: {:.4f}".format(self.HD),
                   "Soft Dice: {:.4f}".format(self.SD),
                   "For positive class:",
                   "TP(sensitivity,recall): {:.4f}".format(self.P_TPR),
                   "PPV(precision): {:.4f}".format(self.P_PPV),
                   "F-1: {:.4f}".format(self.P_F1),
                   "",
                   "For normal class:",
                   "TP(sensitivity,recall): {:.4f}".format(self.N_TPR),
                   "PPV(precision): {:.4f}".format(self.N_PPV),
                   "F-1: {:.4f}".format(self.N_F1)
                   )
        return "\n".join(summary)
