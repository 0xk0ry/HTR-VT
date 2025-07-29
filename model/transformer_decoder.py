import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer decoder."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder for autoregressive text generation.
    Similar to TrOCR decoder architecture.
    """
    
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_seq_len=256):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # Use seq_len first format [S, B, E]
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following standard practice."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_padding_mask(self, tokens, pad_token_id=0):
        """Create padding mask for target sequence."""
        # tokens shape: [B, T]
        return (tokens == pad_token_id).transpose(0, 1)  # [T, B]
    
    def forward(self, memory, tgt, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for training with teacher forcing.
        
        Args:
            memory: Encoder output [L, B, D] where L is sequence length from encoder
            tgt: Target tokens [T, B] where T is target sequence length
            tgt_mask: Subsequent mask for target sequence [T, T]
            tgt_key_padding_mask: Padding mask for target [B, T]
            memory_key_padding_mask: Padding mask for encoder output [B, L]
        
        Returns:
            logits: [T, B, vocab_size]
        """
        # Embedding and positional encoding
        # tgt shape: [T, B] -> [T, B, D]
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Generate subsequent mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        # Transformer decoder
        # memory: [L, B, D], tgt_emb: [T, B, D]
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Output projection
        logits = self.output_projection(output)  # [T, B, vocab_size]
        
        return logits
    
    def decode_step(self, memory, tgt, memory_key_padding_mask=None):
        """
        Single decoding step for inference.
        
        Args:
            memory: Encoder output [L, B, D]
            tgt: Current target sequence [T, B]
            memory_key_padding_mask: Padding mask for encoder output [L, B]
        
        Returns:
            logits: [T, B, vocab_size] - logits for the current sequence
        """
        return self.forward(memory, tgt, memory_key_padding_mask=memory_key_padding_mask)
    
    def nucleus_sampling(self, logits, top_p=0.9, temperature=1.0):
        """
        Nucleus (top-p) sampling for more diverse generation.
        
        Args:
            logits: [B, vocab_size]
            top_p: Cumulative probability threshold
            temperature: Temperature for sampling
        
        Returns:
            tokens: [B] sampled tokens
        """
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create mask for nucleus
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        # Set probabilities outside nucleus to 0
        for batch_idx in range(logits.size(0)):
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            probs[batch_idx, indices_to_remove] = 0
        
        # Renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample from the filtered distribution
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return tokens

    def greedy_decode(self, memory, sos_token_id, eos_token_id, max_len=None, memory_key_padding_mask=None, temperature=1.0, repetition_penalty=1.1):
        """
        Greedy decoding for inference with repetition penalty.
        
        Args:
            memory: Encoder output [L, B, D]
            sos_token_id: Start of sequence token ID
            eos_token_id: End of sequence token ID
            max_len: Maximum generation length
            memory_key_padding_mask: Padding mask for encoder output [L, B]
            temperature: Temperature for sampling (1.0 = no change)
            repetition_penalty: Penalty for repeated tokens (>1.0 = discourage repetition)
        
        Returns:
            sequences: [B, max_decoded_len] - decoded sequences
        """
        if max_len is None:
            max_len = self.max_seq_len
        
        batch_size = memory.size(1)
        device = memory.device
        
        # Initialize with SOS token
        decoded = torch.full((1, batch_size), sos_token_id, dtype=torch.long, device=device)
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_len - 1):
            # Get logits for current sequence
            logits = self.decode_step(memory, decoded, memory_key_padding_mask)
            
            # Apply temperature
            logits_temp = logits[-1] / temperature  # [B, vocab_size]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0 and step > 0:
                # Get tokens that have been generated so far
                generated_tokens = decoded.transpose(0, 1)  # [B, T]
                
                for batch_idx in range(batch_size):
                    if not finished[batch_idx]:
                        # Get unique tokens in this sequence
                        unique_tokens = torch.unique(generated_tokens[batch_idx])
                        
                        # Apply penalty to repeated tokens
                        for token in unique_tokens:
                            if token != sos_token_id:  # Don't penalize SOS token
                                if logits_temp[batch_idx, token] > 0:
                                    logits_temp[batch_idx, token] /= repetition_penalty
                                else:
                                    logits_temp[batch_idx, token] *= repetition_penalty
            
            # Prevent generating consecutive identical tokens
            if step > 0:
                prev_token = decoded[-1]  # [B]
                for batch_idx in range(batch_size):
                    if not finished[batch_idx]:
                        # Strongly discourage repeating the exact same token
                        logits_temp[batch_idx, prev_token[batch_idx]] -= 5.0
            
            # Get next token (greedy from modified logits)
            next_token = logits_temp.argmax(dim=-1)  # [B]
            
            # Append next token
            decoded = torch.cat([decoded, next_token.unsqueeze(0)], dim=0)
            
            # Check for EOS tokens
            finished = finished | (next_token == eos_token_id)
            
            # If all sequences are finished, break
            if finished.all():
                break
        
        return decoded.transpose(0, 1)  # [B, T]
    
    def nucleus_decode(self, memory, sos_token_id, eos_token_id, max_len=None, memory_key_padding_mask=None, temperature=0.8, top_p=0.9, repetition_penalty=1.1):
        """
        Nucleus sampling decoding for more diverse generation.
        
        Args:
            memory: Encoder output [L, B, D]
            sos_token_id: Start of sequence token ID
            eos_token_id: End of sequence token ID
            max_len: Maximum generation length
            memory_key_padding_mask: Padding mask for encoder output [L, B]
            temperature: Temperature for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
        
        Returns:
            sequences: [B, max_decoded_len] - decoded sequences
        """
        if max_len is None:
            max_len = self.max_seq_len
        
        batch_size = memory.size(1)
        device = memory.device
        
        # Initialize with SOS token
        decoded = torch.full((1, batch_size), sos_token_id, dtype=torch.long, device=device)
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_len - 1):
            # Get logits for current sequence
            logits = self.decode_step(memory, decoded, memory_key_padding_mask)
            
            # Current step logits
            current_logits = logits[-1]  # [B, vocab_size]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0 and step > 0:
                generated_tokens = decoded.transpose(0, 1)  # [B, T]
                
                for batch_idx in range(batch_size):
                    if not finished[batch_idx]:
                        unique_tokens = torch.unique(generated_tokens[batch_idx])
                        
                        for token in unique_tokens:
                            if token != sos_token_id:
                                if current_logits[batch_idx, token] > 0:
                                    current_logits[batch_idx, token] /= repetition_penalty
                                else:
                                    current_logits[batch_idx, token] *= repetition_penalty
            
            # Prevent consecutive identical tokens
            if step > 0:
                prev_token = decoded[-1]  # [B]
                for batch_idx in range(batch_size):
                    if not finished[batch_idx]:
                        current_logits[batch_idx, prev_token[batch_idx]] -= 3.0
            
            # Use nucleus sampling
            next_token = self.nucleus_sampling(current_logits, top_p=top_p, temperature=temperature)
            
            # Append next token
            decoded = torch.cat([decoded, next_token.unsqueeze(0)], dim=0)
            
            # Check for EOS tokens
            finished = finished | (next_token == eos_token_id)
            
            # If all sequences are finished, break
            if finished.all():
                break
        
        return decoded.transpose(0, 1)  # [B, T]
    
    def beam_search_decode(self, memory, sos_token_id, eos_token_id, beam_size=5, max_len=None, memory_key_padding_mask=None):
        """
        Beam search decoding for inference.
        
        Args:
            memory: Encoder output [L, B, D] - Note: B should be 1 for beam search
            sos_token_id: Start of sequence token ID
            eos_token_id: End of sequence token ID
            beam_size: Number of beams to keep
            max_len: Maximum generation length
            memory_key_padding_mask: Padding mask for encoder output [L, B]
        
        Returns:
            sequences: [beam_size, max_decoded_len] - top beam_size decoded sequences
            scores: [beam_size] - log probabilities of sequences
        """
        if max_len is None:
            max_len = self.max_seq_len
        
        assert memory.size(1) == 1, "Beam search currently supports batch size 1"
        
        device = memory.device
        vocab_size = self.vocab_size
        
        # Expand memory for beam search
        memory = memory.repeat(1, beam_size, 1)  # [L, beam_size, D]
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.repeat(1, beam_size)
        
        # Initialize beams
        sequences = torch.full((beam_size, 1), sos_token_id, dtype=torch.long, device=device)
        scores = torch.zeros(beam_size, device=device)
        scores[1:] = float('-inf')  # Only first beam is active initially
        
        finished_sequences = []
        finished_scores = []
        
        for step in range(max_len - 1):
            # Current sequences: [beam_size, current_len]
            current_len = sequences.size(1)
            
            # Prepare input for decoder: [current_len, beam_size]
            decoder_input = sequences.transpose(0, 1)
            
            # Get logits: [current_len, beam_size, vocab_size]
            logits = self.decode_step(memory, decoder_input, memory_key_padding_mask)
            
            # Get log probabilities for next token: [beam_size, vocab_size]
            log_probs = F.log_softmax(logits[-1], dim=-1)
            
            # Compute scores for all possible next tokens
            candidate_scores = scores.unsqueeze(1) + log_probs  # [beam_size, vocab_size]
            candidate_scores = candidate_scores.view(-1)  # [beam_size * vocab_size]
            
            # Get top beam_size candidates
            top_scores, top_indices = candidate_scores.topk(beam_size)
            
            # Convert back to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update sequences and scores
            new_sequences = []
            new_scores = []
            
            for i, (beam_idx, token_idx, score) in enumerate(zip(beam_indices, token_indices, top_scores)):
                # Get the sequence from the corresponding beam
                seq = sequences[beam_idx]
                new_seq = torch.cat([seq, token_idx.unsqueeze(0)])
                
                # Check if this sequence ends with EOS
                if token_idx == eos_token_id:
                    finished_sequences.append(new_seq)
                    finished_scores.append(score.item())
                else:
                    new_sequences.append(new_seq)
                    new_scores.append(score)
            
            # Update active beams
            if len(new_sequences) == 0:
                break
            
            # Pad sequences to same length and stack
            max_seq_len = max(len(seq) for seq in new_sequences)
            padded_sequences = []
            for seq in new_sequences:
                if len(seq) < max_seq_len:
                    padded = torch.cat([seq, torch.zeros(max_seq_len - len(seq), dtype=torch.long, device=device)])
                else:
                    padded = seq
                padded_sequences.append(padded)
            
            sequences = torch.stack(padded_sequences)
            scores = torch.tensor(new_scores, device=device)
            
            # Keep only top beam_size sequences
            if len(sequences) > beam_size:
                top_indices = scores.topk(beam_size)[1]
                sequences = sequences[top_indices]
                scores = scores[top_indices]
        
        # Combine finished and unfinished sequences
        all_sequences = finished_sequences + [seq for seq in sequences]
        all_scores = finished_scores + [score.item() for score in scores]
        
        # Sort by score and return top beam_size
        sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
        
        # Pad all sequences to same length
        if all_sequences:
            max_len_final = max(len(seq) for seq in all_sequences)
            final_sequences = []
            final_scores = []
            
            for i in sorted_indices[:beam_size]:
                seq = all_sequences[i]
                score = all_scores[i]
                
                if len(seq) < max_len_final:
                    padded = torch.cat([seq, torch.zeros(max_len_final - len(seq), dtype=torch.long, device=device)])
                else:
                    padded = seq
                
                final_sequences.append(padded)
                final_scores.append(score)
            
            return torch.stack(final_sequences), torch.tensor(final_scores, device=device)
        else:
            # Fallback: return current sequences
            return sequences, scores
