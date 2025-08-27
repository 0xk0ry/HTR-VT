import torch
import torch.distributed as dist
from torch.distributions.uniform import Uniform

import os
import re
import sys
import math
import logging
from copy import deepcopy
from collections import OrderedDict
from typing import Iterable, List, Tuple
from . import vn_tags

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Vietnamese charset helpers (base vs full) for dual-head mode
VIETNAMESE_BASE_CHARACTERS = (
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    '.,!?;: "#&\'()*+-/%=<>@[]^_`{|}~'
    'đĐ'
)

def build_base_charset() -> str:
    return VIETNAMESE_BASE_CHARACTERS


def randint(low, high):
    return int(torch.randint(low, high, (1, )))


def rand_uniform(low, high):
    return float(Uniform(low, high).sample())


def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def update_lr_cos(nb_iter, warm_up_iter, total_iter, max_lr, optimizer, min_lr=1e-7):

    if nb_iter < warm_up_iter:
        current_lr = max_lr * (nb_iter + 1) / (warm_up_iter + 1)
    else:
        current_lr = min_lr + (max_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * nb_iter / (total_iter - warm_up_iter)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


class CTCLabelConverter(object):
    def __init__(self, character):
        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        if len(self.dict) == 87:     # '[' and ']' are not in the test set but in the training and validation sets.
            self.dict['['], self.dict[']'] = 88, 89
        self.character = ['[blank]'] + dict_character

    def encode(self, text):
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text).to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        texts = []
        index = 0

        for l in length:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])) and t[i]<len(self.character):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class DualLabelConverter(object):
    """Converter for base characters + modifier/tone tag supervision.
    encode(texts) returns: (text_base, length_base, tags_mod, tags_tone, per_sample_U)
    decode behaves like CTCLabelConverter for base sequence.
    """
    def __init__(self, base_charset: Iterable[str]):
        chars = list(base_charset)
        self.dict = {ch: i+1 for i, ch in enumerate(chars)}  # 0 reserved for CTC blank
        self.character = ['[blank]'] + chars
        self._cache = {}

    def encode(self, texts: List[str]):
        base_seqs=[]; tags_mod=[]; tags_tone=[]; lengths=[]
        for s in texts:
            if s in self._cache:
                base_str, mod_ids, tone_ids = self._cache[s]
            else:
                base_str, mod_ids, tone_ids = vn_tags.decompose_str(s)
                if len(self._cache) < 10000:
                    self._cache[s] = (base_str, mod_ids, tone_ids)
            base_seqs.append(base_str)
            lengths.append(len(base_str))
            tags_mod.append(torch.LongTensor(mod_ids).to(device))
            tags_tone.append(torch.LongTensor(tone_ids).to(device))
        flat=''.join(base_seqs)
        try:
            text_base=[self.dict[ch] for ch in flat]
        except KeyError as e:
            raise KeyError(f"Missing char {e} in base charset. Update --base-charset if needed.")
        text_base_tensor=torch.IntTensor(text_base).to(device)
        length_tensor=torch.IntTensor(lengths).to(device)
        per_sample_U = length_tensor.clone()
        return text_base_tensor, length_tensor, tags_mod, tags_tone, per_sample_U

    def decode(self, text_index: torch.Tensor, length: torch.Tensor) -> List[str]:
        texts=[]; idx=0
        for l in length:
            l=int(l); t=text_index[idx:idx+l]
            chars=[]
            for i in range(l):
                if t[i] != 0 and (not (i>0 and t[i]==t[i-1])) and t[i] < len(self.character):
                    chars.append(self.character[int(t[i])])
            texts.append(''.join(chars))
            idx += l
        return texts


class Averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class Metric(object):
    def __init__(self, name=''):
        self.name = name
        self.sum = torch.tensor(0.).double()
        self.n = torch.tensor(0.)

    def update(self, val):
        rt = val.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        self.sum += rt.detach().cpu().double()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n.double()


class ModelEma:
    def __init__(self, model, decay=0.9999, device='', resume=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path, mapl=None):
        checkpoint = torch.load(checkpoint_path,map_location=mapl)
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            print("=> Loaded state_dict_ema")
        else:
            print("=> Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model, num_updates=-1):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        if num_updates >= 0:
            _cdecay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        else:
            _cdecay = self.decay

        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * _cdecay + (1. - _cdecay) * model_v)


def format_string_for_wer(str):
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
    str = re.sub('([ \n])+', " ", str).strip()
    return str