import torch
from utils.utils import device


class EncoderDecoderTokenizer(object):
    """
    Tokenizer for Encoder-Decoder model with special tokens.
    Supports SOS, EOS, PAD tokens for autoregressive generation.
    """
    
    def __init__(self, character):
        """
        Initialize tokenizer with character vocabulary.
        
        Args:
            character: String containing all characters in vocabulary
        """
        # Special tokens
        self.PAD_TOKEN = '<pad>'
        self.SOS_TOKEN = '<sos>'
        self.EOS_TOKEN = '<eos>'
        self.UNK_TOKEN = '<unk>'  # For unknown characters
        
        # Build vocabulary
        dict_character = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN] + list(character)
        
        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(dict_character)}
        self.idx_to_char = {idx: char for idx, char in enumerate(dict_character)}
        
        # Special token IDs
        self.pad_token_id = self.char_to_idx[self.PAD_TOKEN]
        self.sos_token_id = self.char_to_idx[self.SOS_TOKEN]
        self.eos_token_id = self.char_to_idx[self.EOS_TOKEN]
        self.unk_token_id = self.char_to_idx[self.UNK_TOKEN]
        
        self.vocab_size = len(dict_character)
        self.character = dict_character
        
        # Debug information
        print(f"DEBUG: Tokenizer created with {self.vocab_size} tokens")
        print(f"DEBUG: Character vocab sample: {character[:50]}...")
        print(f"DEBUG: Special token IDs - PAD:{self.pad_token_id}, SOS:{self.sos_token_id}, EOS:{self.eos_token_id}, UNK:{self.unk_token_id}")
    
    def encode_for_training(self, texts, max_length=None):
        """
        Encode texts for training with teacher forcing.
        Creates input/output pairs by shifting the target sequence.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length (if None, use longest in batch)
        
        Returns:
            tgt_input: [B, T] - decoder input with SOS
            tgt_output: [B, T] - decoder target with EOS
            lengths: [B] - actual sequence lengths (including EOS)
        """
        batch_size = len(texts)
        
        # Convert texts to token indices
        encoded_texts = []
        for text in texts:
            tokens = [self.char_to_idx.get(char, self.unk_token_id) for char in text]
            encoded_texts.append(tokens)
            
            # Debug: Check for unknown characters
            unk_chars = [char for char in text if char not in self.char_to_idx]
            if unk_chars:
                print(f"DEBUG: Unknown characters in '{text[:20]}...': {set(unk_chars)}")
        
        # Determine max length
        if max_length is None:
            max_length = max(len(tokens) for tokens in encoded_texts) + 2  # +2 for SOS/EOS
        
        # Create input and output sequences
        tgt_input = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        tgt_output = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        
        for i, tokens in enumerate(encoded_texts):
            seq_len = min(len(tokens) + 1, max_length - 1)  # +1 for EOS, ensure space for SOS
            
            # Input: [SOS, c1, c2, ..., cN] (or truncated)
            tgt_input[i, 0] = self.sos_token_id
            if seq_len > 1:
                tgt_input[i, 1:seq_len] = torch.tensor(tokens[:seq_len-1])
            
            # Output: [c1, c2, ..., cN, EOS] (or truncated)
            if seq_len > 1:
                tgt_output[i, :seq_len-1] = torch.tensor(tokens[:seq_len-1])
            tgt_output[i, seq_len-1] = self.eos_token_id
            
            lengths[i] = seq_len
        
        return tgt_input.to(device), tgt_output.to(device), lengths.to(device)
    
    def encode_for_inference(self, batch_size=1):
        """
        Create initial input for inference (just SOS tokens).
        
        Args:
            batch_size: Number of sequences to initialize
        
        Returns:
            initial_input: [B, 1] - tensor with SOS tokens
        """
        return torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)
    
    def decode(self, token_sequences):
        """
        Decode token sequences back to text.
        
        Args:
            token_sequences: [B, T] tensor of token indices or list of sequences
        
        Returns:
            texts: List of decoded text strings
        """
        if isinstance(token_sequences, torch.Tensor):
            token_sequences = token_sequences.cpu().numpy()
        
        texts = []
        for seq_idx, tokens in enumerate(token_sequences):
            text_chars = []
            unk_count = 0
            for token_id in tokens:
                # Skip special tokens in output
                if token_id == self.pad_token_id:
                    continue
                elif token_id == self.eos_token_id:
                    break  # Stop at EOS
                elif token_id == self.sos_token_id:
                    continue  # Skip SOS in output
                elif token_id == self.unk_token_id:
                    text_chars.append(self.UNK_TOKEN)
                    unk_count += 1
                elif token_id in self.idx_to_char:
                    text_chars.append(self.idx_to_char[token_id])
                else:
                    text_chars.append(self.UNK_TOKEN)  # Unknown token
                    unk_count += 1
            
            # Debug: warn if too many UNK tokens
            if unk_count > len(text_chars) * 0.5:  # More than 50% UNK
                print(f"DEBUG: Sequence {seq_idx} has {unk_count}/{len(text_chars)} UNK tokens")
                print(f"DEBUG: Token IDs: {tokens[:20]}...")
            
            texts.append(''.join(text_chars))
        
        return texts
    
    def create_padding_mask(self, sequences, pad_token_id=None):
        """
        Create padding mask for sequences.
        
        Args:
            sequences: [B, T] tensor of token indices
            pad_token_id: Padding token ID (if None, use self.pad_token_id)
        
        Returns:
            mask: [B, T] boolean mask (True for padding positions)
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        # Create mask: True for padding tokens
        mask = (sequences == pad_token_id)  # [B, T]
        return mask  # Keep as [B, T] for PyTorch transformer
    
    def get_vocab_info(self):
        """Get vocabulary information."""
        return {
            'vocab_size': self.vocab_size,
            'pad_token_id': self.pad_token_id,
            'sos_token_id': self.sos_token_id,
            'eos_token_id': self.eos_token_id,
            'unk_token_id': self.unk_token_id,
            'character': self.character
        }
