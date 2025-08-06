import unicodedata
def merge_base_and_diacritic(base_seq, diacritic_seq):
    """
    Given a base character sequence and diacritic sequence, return the reconstructed Vietnamese string.
    base_seq: str (e.g. 'cac')
    diacritic_seq: list[int] (e.g. [0, 4, 0])
    """
    # Map diacritic class to combining Unicode
    diacritic_unicode = {
        0: '',
        1: '\u0300', # grave
        2: '\u0309', # hook-above
        3: '\u0303', # tilde
        4: '\u0301', # acute
        5: '\u0323', # dot-below
    }
    chars = []
    for b, d in zip(base_seq, diacritic_seq):
        if d == 0:
            chars.append(b)
        else:
            chars.append(unicodedata.normalize('NFC', b + diacritic_unicode[d]))
    return ''.join(chars)
from torchvision.transforms import ColorJitter
from data import transform as transform
from utils import utils
from torch.utils.data import Dataset
from PIL import Image
import itertools
import os
import skimage
import torch
import numpy as np


def MultiTaskCollate(batch):
    """
    Collate function for multi-task data: handles images, base sequences, and diacritic sequences
    Ensures base and diacritic sequences are padded to the same length
    """
    images, base_labels, diacritic_labels = zip(*batch)
    
    # Convert images to tensors
    image_tensors = [torch.from_numpy(np.array(img, copy=True)) for img in images]
    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
    image_tensors = image_tensors.unsqueeze(1).float()
    
    # Pad diacritic sequences to same length as base sequences
    max_len = max(len(seq) for seq in diacritic_labels)
    padded_diacritic_labels = []
    
    for base_seq, diac_seq in zip(base_labels, diacritic_labels):
        # Ensure diacritic sequence matches base sequence length
        if len(diac_seq) < len(base_seq):
            # Pad with 0 (no diacritic)
            padded_diac = diac_seq + [0] * (len(base_seq) - len(diac_seq))
        else:
            padded_diac = diac_seq[:len(base_seq)]
        
        # Further pad to max_len for batching
        if len(padded_diac) < max_len:
            padded_diac = padded_diac + [0] * (max_len - len(padded_diac))
        
        padded_diacritic_labels.append(padded_diac)
    
    return image_tensors, list(base_labels), padded_diacritic_labels


def SameTrCollate(batch, args):

    images, labels = zip(*batch)
    images = [Image.fromarray(np.uint8(images[i][0] * 255))
              for i in range(len(images))]

    # Apply data augmentations with 90% probability
    if np.random.rand() < 0.5:
        images = [transform.RandomTransform(
            args.proj)(image) for image in images]

    if np.random.rand() < 0.5:
        kernel_h = utils.randint(1, args.dila_ero_max_kernel + 1)
        kernel_w = utils.randint(1, args.dila_ero_max_kernel + 1)
        if utils.randint(0, 2) == 0:
            images = [transform.Erosion((kernel_w, kernel_h), args.dila_ero_iter)(
                image) for image in images]
        else:
            images = [transform.Dilation((kernel_w, kernel_h), args.dila_ero_iter)(
                image) for image in images]

    if np.random.rand() < 0.5:
        images = [ColorJitter(args.jitter_brightness, args.jitter_contrast, args.jitter_saturation,
                              args.jitter_hue)(image) for image in images]

    # Convert images to tensors

    image_tensors = [torch.from_numpy(
        np.array(image, copy=True)) for image in images]
    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
    image_tensors = image_tensors.unsqueeze(1).float()
    image_tensors = image_tensors / 255.
    return image_tensors, labels


class myLoadDS(Dataset):
    def __init__(self, flist, dpath, img_size=[512, 32], ralph=None, fmin=True, mln=None):
        self.fns = get_files(flist, dpath)
        self.tlbls = get_labels(self.fns)
        self.img_size = img_size

        # Vietnamese base alphabet (no diacritics)
        self.base_alphabet = (
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789'
            '.,!?;: "#&\'()*+-/%=<>@[]^_`{|}~'
            'ăâêôơưđ'
            'ĂÂÊÔƠƯĐ'
        )
        # Diacritic classes: 0=no, 1=grave, 2=hook-above, 3=perispomeni, 4=acute, 5=dot-below
        self.diacritic_map = {
            '\u0300': 1, # grave
            '\u0309': 2, # hook-above
            '\u0303': 3, # perispomeni (tilde)
            '\u0301': 4, # acute
            '\u0323': 5, # dot-below
        }

        if ralph is None:
            alph = get_alphabet(self.tlbls)
            self.ralph = dict(zip(alph.values(), alph.keys()))
            self.alph = alph
        else:
            self.ralph = ralph
        self.ralph = {
            idx: char for idx, char in enumerate(
                self.base_alphabet
            )
        }
        if mln is not None:
            filt = [len(x) <= mln if fmin else len(x) >= mln for x in self.tlbls]
            self.tlbls = np.asarray(self.tlbls)[filt].tolist()
            self.fns = np.asarray(self.fns)[filt].tolist()

        # Precompute decomposed labels
        self.base_labels = []
        self.diacritic_labels = []
        for lbl in self.tlbls:
            base_seq, diac_seq = decompose_vietnamese(lbl)
            self.base_labels.append(base_seq)
            self.diacritic_labels.append(diac_seq)

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        timgs = get_images(self.fns[index], self.img_size[0], self.img_size[1])
        timgs = timgs.transpose((2, 0, 1))
        # Return image, base-char sequence, diacritic sequence
        return (timgs, self.base_labels[index], self.diacritic_labels[index])
import unicodedata
def decompose_vietnamese(text):
    # Returns (base_char_seq, diacritic_seq) for a given Vietnamese string
    base_seq = []
    diac_seq = []
    for char in text:
        norm = unicodedata.normalize('NFD', char)
        base = norm[0]
        diacritic = 0
        for c in norm[1:]:
            if c in ['\u0300', '\u0309', '\u0303', '\u0301', '\u0323']:
                # Map to diacritic class
                diacritic = {
                    '\u0300': 1, # grave
                    '\u0309': 2, # hook-above
                    '\u0303': 3, # perispomeni
                    '\u0301': 4, # acute
                    '\u0323': 5, # dot-below
                }[c]
        base_seq.append(base)
        diac_seq.append(diacritic)
    return ''.join(base_seq), diac_seq


def get_files(nfile, dpath):
    fnames = open(nfile, 'r').readlines()
    fnames = [dpath + x.strip() for x in fnames]
    return fnames


def npThum(img, max_w, max_h):
    x, y = np.shape(img)[:2]

    y = min(int(y * max_h / x), max_w)
    x = max_h

    img = np.array(Image.fromarray(img).resize((y, x)))
    return img


def get_images(fname, max_w=500, max_h=500, nch=1):  # args.max_w args.max_h args.nch

    try:

        image_data = np.array(Image.open(fname).convert('L'))
        image_data = npThum(image_data, max_w, max_h)
        image_data = skimage.img_as_float32(image_data)

        h, w = np.shape(image_data)[:2]
        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)

        if nch == 3 and image_data.shape[2] != 3:
            image_data = np.tile(image_data, 3)

        image_data = np.pad(image_data, ((0, 0), (0, max_w - np.shape(image_data)[1]), (0, 0)), mode='constant',
                            constant_values=(1.0))

    except IOError as e:
        print('Could not read:', fname, ':', e)

    return image_data


def get_labels(fnames):
    labels = []
    for id, image_file in enumerate(fnames):
        fn = os.path.splitext(image_file)[0] + '.txt'
        lbl = open(fn, 'r').read()
        lbl = ' '.join(lbl.split())  # remove linebreaks if present

        labels.append(lbl)

    return labels


def get_alphabet(labels):
    coll = ''.join(labels)
    unq = sorted(list(set(coll)))
    unq = [''.join(i) for i in itertools.product(unq, repeat=1)]
    alph = dict(zip(unq, range(len(unq))))

    return alph


def cycle_dpp(iterable):
    epoch = 0
    iterable.sampler.set_epoch(epoch)
    while True:
        for x in iterable:
            yield x
        epoch += 1
        iterable.sampler.set_epoch(epoch)


def cycle_data(iterable):
    while True:
        for x in iterable:
            yield x