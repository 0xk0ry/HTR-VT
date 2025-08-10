import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions.uniform import Uniform

import os
import re
import sys
import math
import logging
from copy import deepcopy
from collections import OrderedDict
import unicodedata

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


# ----------------------
# CTC posteriors and tone helpers
# ----------------------

def ctc_posteriors(log_probs: torch.Tensor, y_indices: torch.Tensor, blank: int = 0) -> torch.Tensor:
    """
    Compute CTC state posteriors and return per-label posteriors gamma[t, j].

    Args:
        log_probs: (T, C) log-softmax over classes for one sample.
        y_indices: (U,) target indices matching the logits' class space (no blanks).
        blank: index of the blank class in logits.
    Returns:
        gamma: (T, U) posterior over target positions j at each time t.
    """
    device = log_probs.device
    T, C = log_probs.shape
    U = int(y_indices.numel())
    if U == 0:
        return log_probs.new_zeros((T, 0))

    # Build extended label sequence with blanks interleaved: length S=2U+1
    S = 2 * U + 1
    ext = torch.empty(S, dtype=torch.long, device=device)
    ext[0::2] = blank
    ext[1::2] = y_indices

    # Emissions: e[t, k] = log_probs[t, ext[k]]
    e = log_probs[:, ext]  # (T, S)

    neg_inf = torch.tensor(-1e9, device=device)

    # Forward (alpha) in log-space
    alpha = e.new_full((T, S), neg_inf)
    alpha[0, 0] = e[0, 0]
    if S > 1:
        alpha[0, 1] = e[0, 1]
    for t in range(1, T):
        # stay
        terms = alpha[t - 1]
        # move 1
        terms1 = torch.roll(alpha[t - 1], shifts=1, dims=0)
        terms1[0] = neg_inf
        # move 2 (skip) allowed if current symbol is not blank and not equal to symbol two back
        terms2 = torch.roll(alpha[t - 1], shifts=2, dims=0)
        terms2[:2] = neg_inf
        # mask for allowed skip transitions
        not_blank = (ext != blank)
        neq_prev2 = torch.ones(S, dtype=torch.bool, device=device)
        neq_prev2[2:] = ext[2:] != ext[:-2]
        allow_skip = not_blank & neq_prev2
        combine = torch.stack([
            terms,
            terms1,
            torch.where(allow_skip, terms2, neg_inf.expand_as(terms2))
        ], dim=0)
        alpha[t] = torch.logsumexp(combine, dim=0) + e[t]

    # Backward (beta) in log-space
    beta = e.new_full((T, S), neg_inf)
    beta[T - 1, S - 1] = e[T - 1, S - 1]
    if S > 1:
        beta[T - 1, S - 2] = e[T - 1, S - 2]
    for t in range(T - 2, -1, -1):
        # stay
        terms = beta[t + 1]
        # move 1
        terms1 = torch.roll(beta[t + 1], shifts=-1, dims=0)
        terms1[-1] = neg_inf
        # move 2 (skip)
        terms2 = torch.roll(beta[t + 1], shifts=-2, dims=0)
        terms2[-2:] = neg_inf
        not_blank = (ext != blank)
        neq_next2 = torch.ones(S, dtype=torch.bool, device=device)
        neq_next2[:-2] = ext[:-2] != ext[2:]
        allow_skip = not_blank & neq_next2
        combine = torch.stack([
            terms,
            terms1,
            torch.where(allow_skip, terms2, neg_inf.expand_as(terms2))
        ], dim=0)
        beta[t] = torch.logsumexp(combine, dim=0) + e[t]

    # Total log-likelihood Z
    logZ = torch.logsumexp(alpha[T - 1, max(0, S - 2):S], dim=0)
    # State posteriors
    post = torch.exp(alpha + beta - logZ)  # (T, S)
    # Map to label positions (odd indices in ext)
    label_posts = post[:, 1::2]  # (T, U)
    # Normalize small numerical issues per time
    label_posts = label_posts / (label_posts.sum(dim=1, keepdim=True) + 1e-8)
    return label_posts


_VIET_BASE_VOWELS = set(list("aăâeêioôơuưyAĂÂEÊIOÔƠUƯY"))


def is_vietnamese_vowel(ch: str) -> bool:
    if not ch:
        return False
    d = unicodedata.normalize('NFD', ch)
    base = ''.join([c for c in d if unicodedata.category(c) != 'Mn'])
    return any(b in _VIET_BASE_VOWELS for b in base)


def tone_of_char(ch: str) -> int:
    """
    Return tone id: 0=NONE, 1=ACUTE, 2=GRAVE, 3=HOOK, 4=TILDE, 5=DOT
    """
    if not ch:
        return 0
    d = unicodedata.normalize('NFD', ch)
    marks = {ord(c) for c in d if unicodedata.category(c) == 'Mn'}
    if 0x0301 in marks:
        return 1  # ACUTE
    if 0x0300 in marks:
        return 2  # GRAVE
    if 0x0309 in marks:
        return 3  # HOOK ABOVE
    if 0x0303 in marks:
        return 4  # TILDE
    if 0x0323 in marks:
        return 5  # DOT BELOW
    return 0


def is_english_label(label: str) -> bool:
    if not label:
        return True
    for ch in label:
        if ch in 'đĐ':
            return False
        if is_vietnamese_vowel(ch):
            return False
        # Any combining mark indicates Vietnamese tone/diacritic
        if any(unicodedata.category(c) == 'Mn' for c in unicodedata.normalize('NFD', ch)):
            return False
    return True


def vowel_indices_from_converter(converter) -> list:
    idxs = []
    for ch, idx in converter.dict.items():  # idx in [1..]
        try:
            if is_vietnamese_vowel(ch):
                idxs.append(idx)
        except Exception:
            continue
    return idxs


def apply_tone_to_char(ch: str, tone_id: int) -> str:
    """
    Apply the given tone to a Vietnamese vowel character.
    tone_id: 0=NONE, 1=ACUTE, 2=GRAVE, 3=HOOK, 4=TILDE, 5=DOT
    For non-vowel characters, return as-is.
    """
    if not ch or not is_vietnamese_vowel(ch):
        return ch
    # Decompose and remove any existing tone marks
    d = unicodedata.normalize('NFD', ch)
    # Combining marks for tones
    tone_marks = {0x0301, 0x0300, 0x0309, 0x0303, 0x0323}
    base_plus = ''.join(c for c in d if not (unicodedata.category(c) == 'Mn' and ord(c) in tone_marks))
    # Add new mark if needed
    tone_map = {1: '\u0301', 2: '\u0300', 3: '\u0309', 4: '\u0303', 5: '\u0323'}
    if tone_id in tone_map:
        base_plus = base_plus + tone_map[tone_id]
    # Recompose
    return unicodedata.normalize('NFC', base_plus)


# Alias as requested name
def is_vowel_char(ch: str) -> bool:
    return is_vietnamese_vowel(ch)


def contains_vietnamese(text: str) -> bool:
    """
    Quick language gate: True if any Vietnamese vowel/đ or combining tone mark present.
    """
    if not text:
        return False
    tone_marks = {0x0301, 0x0300, 0x0309, 0x0303, 0x0323}
    for ch in text:
        if ch in 'đĐ':
            return True
        if is_vietnamese_vowel(ch):
            return True
        for c in unicodedata.normalize('NFD', ch):
            if unicodedata.category(c) == 'Mn' and ord(c) in tone_marks:
                return True
    return False


def ctc_posteriors_batched(log_probs: torch.Tensor, targets: list, blank: int = 0):
    """
    Batched wrapper for CTC posteriors.

    Args:
        log_probs: (B, T, C) log-softmax over classes per sample.
        targets: list of length B, each is 1D LongTensor of target indices (no blanks).
        blank: index of the blank class.
    Returns:
        List[Tensor]: per-sample gamma of shape (T, U_i).
    """
    B, T, C = log_probs.shape
    assert len(targets) == B, "targets list length must equal batch size"
    gammas = []
    for b in range(B):
        gammas.append(ctc_posteriors(log_probs[b], targets[b].to(log_probs.device), blank=blank))
    return gammas


def aggregate_tone_over_span(tone_logits: torch.Tensor, gate: torch.Tensor, t1: int, t2: int, margin_kappa: float = 0.3) -> int:
    """
    Aggregate tone predictions over a frame span [t1, t2) using a gate and select tone by margin over NONE.

    Args:
        tone_logits: (T, 6)
        gate: (T,) with 0/1 weights per frame
        t1, t2: span start (inclusive) and end (exclusive)
        margin_kappa: minimum margin avg_prob[tone] - avg_prob[NONE] to accept a tone
    Returns:
        tone_id in {0..5}
    """
    t1 = max(0, int(t1))
    t2 = min(tone_logits.size(0), int(t2))
    if t2 <= t1:
        return 0
    probs = F.softmax(tone_logits[t1:t2], dim=-1)  # (L, 6)
    g = gate[t1:t2].float()
    denom = g.sum()
    if float(denom.item()) < 1e-6:
        g = torch.ones_like(g)
        denom = g.sum()
    avg = (probs * g.view(-1, 1)).sum(dim=0) / denom
    # Compare tones 1..5 against NONE (0)
    if avg.numel() < 6:
        return 0
    best_tone = int(avg[1:].argmax().item()) + 1
    margin = float(avg[best_tone] - avg[0])
    return best_tone if margin >= margin_kappa else 0