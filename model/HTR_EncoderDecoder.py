import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

from model import resnet18
from model.transformer_decoder import TransformerDecoder
from model.HTR_VT import get_2d_sincos_pos_embed, LayerNorm, Block


class HTR_EncoderDecoder(nn.Module):
    """
    Handwritten Text Recognition model with Encoder-Decoder architecture.
    
    - Encoder: ResNet18 + ViT blocks (reuse from original HTR_VT)
    - Decoder: Transformer decoder with cross-attention
    - Loss: CrossEntropyLoss (instead of CTC)
    """
    
    def __init__(self,
                 vocab_size=80,
                 img_size=[512, 32],
                 patch_size=[8, 32],
                 embed_dim=768,
                 encoder_depth=4,
                 encoder_num_heads=6,
                 decoder_layers=6,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 max_seq_len=256,
                 dropout=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # ========== ENCODER (reuse from HTR_VT) ==========
        self.layer_norm = LayerNorm()
        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # Positional embedding for encoder
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=False  # fixed sin-cos embedding
        )
        
        # Encoder transformer blocks
        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, encoder_num_heads, self.num_patches,
                  mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)
        ])
        
        self.encoder_norm = norm_layer(embed_dim, elementwise_affine=True)
        
        # ========== DECODER ==========
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=embed_dim,
            nhead=decoder_num_heads,
            num_layers=decoder_layers,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize other weights
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def encode(self, x):
        """
        Encode input image to feature sequence.
        
        Args:
            x: Input image [B, C, H, W]
        
        Returns:
            memory: Encoded features [L, B, D] where L = num_patches
        """
        # Normalize and extract patches
        x = self.layer_norm(x)
        x = self.patch_embed(x)  # [B, C, W, H]
        
        b, c, w, h = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # [B, L, D]
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.encoder_norm(x)
        
        # Convert to [L, B, D] for transformer
        memory = x.transpose(0, 1)
        
        return memory
    
    def decode(self, memory, tgt, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Decode target sequence given encoder memory.
        
        Args:
            memory: Encoder output [L, B, D]
            tgt: Target tokens [T, B]
            tgt_mask: Causal mask for target sequence
            tgt_key_padding_mask: Padding mask for target [B, T]
            memory_key_padding_mask: Padding mask for encoder output [B, L]
        
        Returns:
            logits: [T, B, vocab_size]
        """
        return self.decoder(
            memory=memory,
            tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
    
    def forward(self, x, tgt=None, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass for training with teacher forcing.
        
        Args:
            x: Input image [B, C, H, W]
            tgt: Target tokens [T, B] for teacher forcing
            tgt_mask: Causal mask for target sequence
            tgt_key_padding_mask: Padding mask for target [B, T]
        
        Returns:
            logits: [T, B, vocab_size] if tgt is provided, else memory [L, B, D]
        """
        # Encode
        memory = self.encode(x)
        
        # If no target, return encoder output (for inference initialization)
        if tgt is None:
            return memory
        
        # Decode
        logits = self.decode(
            memory=memory,
            tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return logits
    
    def generate(self, x, sos_token_id, eos_token_id, max_len=None, method='greedy', beam_size=5, temperature=1.0, repetition_penalty=1.1, top_p=0.9):
        """
        Generate text from image using different decoding strategies.
        
        Args:
            x: Input image [B, C, H, W]
            sos_token_id: Start of sequence token ID
            eos_token_id: End of sequence token ID
            max_len: Maximum generation length
            method: 'greedy', 'nucleus', or 'beam_search'
            beam_size: Beam size for beam search
            temperature: Temperature for sampling
            repetition_penalty: Penalty for repeated tokens
            top_p: Nucleus sampling threshold
        
        Returns:
            sequences: Generated token sequences
            scores: Sequence scores (for beam search)
        """
        if max_len is None:
            max_len = self.max_seq_len
        
        # Encode
        memory = self.encode(x)
        
        if method == 'greedy':
            sequences = self.decoder.greedy_decode(
                memory=memory,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id,
                max_len=max_len,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
            return sequences, None
        
        elif method == 'nucleus':
            sequences = self.decoder.nucleus_decode(
                memory=memory,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id,
                max_len=max_len,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            return sequences, None
        
        elif method == 'beam_search':
            # Note: beam search currently supports batch_size=1
            if x.size(0) != 1:
                raise ValueError("Beam search currently supports batch size 1")
            
            sequences, scores = self.decoder.beam_search_decode(
                memory=memory,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id,
                beam_size=beam_size,
                max_len=max_len
            )
            return sequences, scores
        
        else:
            raise ValueError(f"Unknown decoding method: {method}")


def create_encoder_decoder_model(vocab_size, img_size, **kwargs):
    """
    Create HTR Encoder-Decoder model with default hyperparameters.
    """
    model = HTR_EncoderDecoder(
        vocab_size=vocab_size,
        img_size=img_size,
        patch_size=(4, 64),
        embed_dim=768,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_layers=6,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
