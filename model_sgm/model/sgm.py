import torch, torch.nn as nn, torch.nn.functional as F

def build_sgm_vocab(ctc_chars):
    # include BOS/EOS/MASK distinct from CTC’s blank
    specials = ["[BOS]","[EOS]","[MASK]"]
    itos = specials + list(ctc_chars)
    stoi = {ch:i for i,ch in enumerate(itos)}
    return itos, stoi

def make_sgm_batch(labels, stoi, mask_rate=0.15):
    """
    labels: list[str] raw GT per line (NFC normalized)
    returns:
      txt_ids: (B, N) int64 with [BOS] ... [EOS], some chars replaced by [MASK]
      tgt_ids: (B, N) int64 ground-truth char ids (same length), -100 where not masked (ignored by CE)
      mask_positions: bool (B, N)
    """
    BOS, EOS, MASK = stoi["[BOS]"], stoi["[EOS]"], stoi["[MASK]"]
    B = len(labels)
    txt_ids, tgt_ids, mask_pos = [], [], []
    for s in labels:
        seq = [BOS] + [stoi[c] for c in s if c in stoi] + [EOS]
        n  = len(seq)
        m  = max(1, int((n-2) * mask_rate))   # don’t count BOS/EOS
        mask_idx = set()
        # sample positions in [1, n-2]
        while len(mask_idx) < m:
            i = 1 + int(torch.randint(0, n-2, (1,)).item())
            mask_idx.add(i)
        cur_txt, cur_tgt, cur_mask = [], [], []
        for i, tok in enumerate(seq):
            if i in mask_idx:
                cur_txt.append(MASK); cur_tgt.append(tok); cur_mask.append(True)
            else:
                cur_txt.append(tok);  cur_tgt.append(-100); cur_mask.append(False)
        txt_ids.append(cur_txt); tgt_ids.append(cur_tgt); mask_pos.append(cur_mask)
    # pad to max length in batch
    maxN = max(len(x) for x in txt_ids)
    def pad2d(arr, pad_val):
        return [row + [pad_val]*(maxN-len(row)) for row in arr]
    txt_ids = torch.tensor(pad2d(txt_ids, EOS), dtype=torch.long)
    tgt_ids = torch.tensor(pad2d(tgt_ids, -100), dtype=torch.long)
    mask_positions = torch.tensor(pad2d(mask_pos, False), dtype=torch.bool)
    return txt_ids, tgt_ids, mask_positions


class SGMHead(nn.Module):
    """
    Training-only Semantic Guidance Module
    - Embed GT char sequence (with [BOS]/[EOS]/[MASK])
    - Bi-directional self-attn over text (tiny Transformer)
    - Cross-attn from text tokens to visual encoder tokens (keys/values)
    - Predict masked characters (CE loss over masked positions only)
    """
    def __init__(self, vocab_size: int, d_vis: int, d_sgm: int = 256,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, d_sgm)
        self.pos_emb  = nn.Embedding(2048, d_sgm)   # enough for typical line lengths
        enc_layer = nn.TransformerEncoderLayer(d_model=d_sgm, nhead=num_heads,
                                               dim_feedforward=d_sgm*4,
                                               dropout=dropout, batch_first=True)
        self.txt_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # project visual tokens to K,V for cross-attn; text to Q
        self.q_proj = nn.Linear(d_sgm, d_sgm)
        self.k_proj = nn.Linear(d_vis, d_sgm)
        self.v_proj = nn.Linear(d_vis, d_sgm)
        self.out    = nn.Linear(d_sgm, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale   = d_sgm ** -0.5

    def forward(self, vis_tokens, txt_ids, mask_positions):
        """
        vis_tokens: (B, L, d_vis)  from encoder
        txt_ids:    (B, N)         GT text with [MASK] at selected positions
        mask_positions: boolean (B, N) True where token was masked (compute loss there)
        """
        B, N = txt_ids.shape
        device = txt_ids.device
        pos = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        t = self.char_emb(txt_ids) + self.pos_emb(pos)              # (B,N,d_sgm)
        t = self.txt_encoder(t)                                     # (B,N,d_sgm)

        Q = self.q_proj(t)                                          # (B,N,d_sgm)
        K = self.k_proj(vis_tokens)                                  # (B,L,d_sgm)
        V = self.v_proj(vis_tokens)                                  # (B,L,d_sgm)

        attn = torch.softmax(torch.matmul(Q, K.transpose(1, 2)) * self.scale, dim=-1) # (B,N,L)
        ctx  = torch.matmul(attn, V)                                 # (B,N,d_sgm)

        logits = self.out(self.dropout(ctx))                         # (B,N,V)
        return logits, mask_positions
