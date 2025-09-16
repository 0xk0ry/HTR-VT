import torch
import numpy as np


def build_smtr_vocab(chars):
    # chars: iterable of your base alphabet (the same one used by CTC)
    # We add [B] (begin) and [E] (end) as special classes for next/prev
    specials = ["[B]", "[E]"]
    itos = specials + list(chars)
    stoi = {ch: i for i, ch in enumerate(itos)}
    return itos, stoi

def _rand_replace_token(token_id, base_ids, p=0.15):
    # replace with a random *different* base char id with prob p
    if np.random.rand() >= p: return token_id
    if token_id not in base_ids or len(base_ids) <= 1: return token_id
    # choose a different one
    choices = [bid for bid in base_ids if bid != token_id]
    return int(np.random.choice(choices))

def make_smtr_batch(labels, stoi, ls=5, max_substrings=24, rand_replace_p=0.15):
    """
    labels: list[str] ground-truth strings of the batch
    stoi: dict mapping char -> id (contains [B],[E] + base chars)
    returns tensors:
      substr   : (B, S, ls) int64
      tgt_next : (B, S)     int64
      tgt_prev : (B, S)     int64
    S = fixed number of sampled substrings per line (cap for compute)
    Strategy: pick random start s, substring = y[s : s+ls] (pad on the right with [E] if short),
              NEXT label = y[s+ls] if exists else [E],
              PREV label = y[s-1]  if s>0    else [B].
    """
    B = len(labels)
    S = max_substrings
    substr = np.zeros((B, S, ls), dtype=np.int64)
    tgt_n  = np.zeros((B, S),     dtype=np.int64)
    tgt_p  = np.zeros((B, S),     dtype=np.int64)

    # base char ids (exclude specials for RT sampling)
    base_ids = [i for ch, i in stoi.items() if ch not in ("[B]", "[E]")]

    for b, text in enumerate(labels):
        # map chars to ids (unknowns are skipped; you should ensure ralph covers all)
        ids = [stoi.get(ch, None) for ch in text]
        ids = [i for i in ids if i is not None]
        n = len(ids)
        if n == 0:
            # degenerate; fill with begin/end to avoid NaNs
            substr[b, :, :] = stoi["[B]"]
            tgt_n[b, :] = stoi["[E]"]
            tgt_p[b, :] = stoi["[B]"]
            continue

        # sample S start positions (with replacement if needed)
        starts = np.random.randint(0, max(1, n), size=S)
        for s_idx, s in enumerate(starts):
            # substring tokens
            win = ids[s : s + ls]
            # pad to ls with end tokens (right padding is okay)
            if len(win) < ls:
                win = win + [stoi["[E]"]] * (ls - len(win))

            # Regularization Training (replace one token randomly)
            if rand_replace_p > 0.0:
                pos = np.random.randint(0, ls)
                win[pos] = _rand_replace_token(win[pos], base_ids, p=rand_replace_p)

            substr[b, s_idx, :] = np.array(win, dtype=np.int64)

            # NEXT target: char right after the window
            nxt = ids[s + ls] if (s + ls) < n else stoi["[E]"]
            # PREV target: char before the window
            prv = ids[s - 1]  if s > 0 else stoi["[B]"]
            tgt_n[b, s_idx] = nxt
            tgt_p[b, s_idx] = prv

    # to torch tensors
    return (
        torch.from_numpy(substr),  # (B,S,ls)
        torch.from_numpy(tgt_n),   # (B,S)
        torch.from_numpy(tgt_p),   # (B,S)
    )


class SMTRHead(torch.nn.Module):
    """
    Training-only head for substring next/prev prediction.
    - char_emb: embeds GT substring chars
    - a tiny pooling to build a query (per substring)
    - cross-attn from query to encoder tokens to pick visual context
    - two classifiers: NEXT and PREV
    """
    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.char_emb = torch.nn.Embedding(vocab_size, d_model)
        self.pool = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, d_model)
        )
        self.q_proj_n = torch.nn.Linear(d_model, d_model)
        self.q_proj_p = torch.nn.Linear(d_model, d_model)
        self.k_proj   = torch.nn.Linear(d_model, d_model)
        self.v_proj   = torch.nn.Linear(d_model, d_model)
        self.out_next = torch.nn.Linear(d_model, vocab_size)
        self.out_prev = torch.nn.Linear(d_model, vocab_size)
        self.dropout  = torch.nn.Dropout(dropout)
        self.scale = d_model ** -0.5

    def forward(self, enc_tokens, substr_idx):
        """
        enc_tokens: (B, L, D) encoder features from HTR-VT
        substr_idx: (B, S, ls) substring char indices
        returns: logits_next, logits_prev  both shape (B, S, V)
        """
        B, L, D = enc_tokens.shape
        B2, S, ls = substr_idx.shape
        assert B == B2, "Batch mismatch"

        # embed and mean-pool the substring (simple, stable)
        emb = self.char_emb(substr_idx)      # (B, S, ls, D)
        pooled = emb.mean(dim=2)             # (B, S, D)

        # queries for NEXT and PREV
        qn = self.q_proj_n(pooled)           # (B, S, D)
        qp = self.q_proj_p(pooled)           # (B, S, D)

        # keys/values from encoder
        K  = self.k_proj(enc_tokens)         # (B, L, D)
        V  = self.v_proj(enc_tokens)         # (B, L, D)

        # (B, S, D) @ (B, D, L) = (B, S, L)
        attn_n = torch.softmax(torch.matmul(qn, K.transpose(1, 2)) * self.scale, dim=-1)
        ctx_n  = torch.matmul(attn_n, V)     # (B, S, D)

        attn_p = torch.softmax(torch.matmul(qp, K.transpose(1, 2)) * self.scale, dim=-1)
        ctx_p  = torch.matmul(attn_p, V)     # (B, S, D)

        # classify
        logits_next = self.out_next(self.dropout(ctx_n))  # (B, S, V)
        logits_prev = self.out_prev(self.dropout(ctx_p))  # (B, S, V)
        return logits_next, logits_prev
