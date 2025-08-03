"""
Utilities to convert between CTC-based and Encoder-Decoder training setups.
"""

import torch
from utils.utils import CTCLabelConverter
from utils.encoder_decoder_tokenizer import EncoderDecoderTokenizer


def convert_ctc_to_encoder_decoder_alphabet(ctc_alphabet):
    """
    Convert CTC alphabet to encoder-decoder alphabet.
    CTC alphabet includes [blank] at the beginning.

    Args:
        ctc_alphabet: List of characters from CTC converter

    Returns:
        ed_alphabet: String of characters for encoder-decoder
    """
    # Remove [blank] token if present
    if ctc_alphabet[0] == '[blank]':
        ed_alphabet = ''.join(ctc_alphabet[1:])
    else:
        ed_alphabet = ''.join(ctc_alphabet)

    return ed_alphabet


def create_encoder_decoder_tokenizer_from_ctc(ctc_converter):
    """
    Create EncoderDecoderTokenizer from existing CTCLabelConverter.

    Args:
        ctc_converter: CTCLabelConverter instance

    Returns:
        ed_tokenizer: EncoderDecoderTokenizer instance
    """
    ed_alphabet = convert_ctc_to_encoder_decoder_alphabet(
        ctc_converter.character)
    ed_tokenizer = EncoderDecoderTokenizer(ed_alphabet)
    return ed_tokenizer


def get_alphabet_from_file_list(file_list_path, data_path=None):
    """
    Extract alphabet from dataset file list.

    Args:
        file_list_path: Path to .ln file containing image/text pairs
        data_path: Path to data directory (not used, kept for compatibility)

    Returns:
        alphabet: String containing all unique characters
    """
    alphabet_set = set()

    with open(file_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            text = parts[1]
            alphabet_set.update(text)

    # Sort for consistent ordering
    alphabet = ''.join(sorted(alphabet_set))
    return alphabet


def validate_tokenizer_compatibility(tokenizer1, tokenizer2):
    """
    Check if two tokenizers are compatible (same vocabulary).

    Args:
        tokenizer1: First tokenizer
        tokenizer2: Second tokenizer

    Returns:
        compatible: Boolean indicating compatibility
        differences: List of differences if any
    """
    differences = []

    # Check vocab sizes
    if tokenizer1.vocab_size != tokenizer2.vocab_size:
        differences.append(
            f"Vocab sizes differ: {tokenizer1.vocab_size} vs {tokenizer2.vocab_size}")

    # Check character sets (excluding special tokens for encoder-decoder)
    if hasattr(tokenizer1, 'character') and hasattr(tokenizer2, 'character'):
        chars1 = set(tokenizer1.character)
        chars2 = set(tokenizer2.character)

        # For encoder-decoder, remove special tokens from comparison
        if hasattr(tokenizer2, 'PAD_TOKEN'):
            special_tokens = {tokenizer2.PAD_TOKEN, tokenizer2.SOS_TOKEN,
                              tokenizer2.EOS_TOKEN, tokenizer2.UNK_TOKEN}
            chars2 = chars2 - special_tokens

        # For CTC, remove blank token
        if '[blank]' in chars1:
            chars1 = chars1 - {'[blank]'}

        if chars1 != chars2:
            only_in_1 = chars1 - chars2
            only_in_2 = chars2 - chars1
            if only_in_1:
                differences.append(
                    f"Characters only in tokenizer1: {only_in_1}")
            if only_in_2:
                differences.append(
                    f"Characters only in tokenizer2: {only_in_2}")

    compatible = len(differences) == 0
    return compatible, differences


def create_dataset_specific_tokenizer(dataset_name, train_data_list, data_path):
    """
    Create tokenizer for specific dataset.

    Args:
        dataset_name: Name of dataset ('IAM', 'READ', 'LAM')
        train_data_list: Path to training data list
        data_path: Path to data directory

    Returns:
        tokenizer: EncoderDecoderTokenizer instance
        alphabet: Extracted alphabet string
    """
    alphabet = get_alphabet_from_file_list(train_data_list, data_path)

    # Dataset-specific adjustments if needed
    if dataset_name.upper() == 'IAM':
        # IAM specific processing if needed
        pass
    elif dataset_name.upper() == 'READ':
        # READ2016 specific processing if needed
        pass
    elif dataset_name.upper() == 'LAM':
        # LAM specific processing if needed
        pass

    tokenizer = EncoderDecoderTokenizer(alphabet)

    return tokenizer, alphabet


def compare_model_outputs(ctc_model, ed_model, sample_input, ctc_converter, ed_tokenizer):
    """
    Compare outputs between CTC model and Encoder-Decoder model for debugging.

    Args:
        ctc_model: CTC-based model
        ed_model: Encoder-Decoder model
        sample_input: Sample input image tensor
        ctc_converter: CTC converter
        ed_tokenizer: Encoder-Decoder tokenizer

    Returns:
        comparison: Dictionary with comparison results
    """
    device = sample_input.device

    with torch.no_grad():
        # CTC model output
        ctc_output = ctc_model(sample_input)  # [B, L, vocab_size]
        ctc_preds = ctc_output.argmax(dim=-1)  # [B, L]
        ctc_texts = ctc_converter.decode(
            ctc_preds[0].cpu(), [ctc_preds.shape[1]])

        # Encoder-Decoder model encoding
        ed_memory = ed_model.encode(sample_input)  # [L, B, D]

        # Simple greedy decoding for comparison
        ed_sequences, _ = ed_model.generate(
            sample_input,
            sos_token_id=ed_tokenizer.sos_token_id,
            eos_token_id=ed_tokenizer.eos_token_id,
            method='greedy'
        )
        ed_texts = ed_tokenizer.decode(ed_sequences)

    comparison = {
        'ctc_output_shape': ctc_output.shape,
        'ed_memory_shape': ed_memory.shape,
        'ctc_text': ctc_texts[0] if ctc_texts else '',
        'ed_text': ed_texts[0] if ed_texts else '',
        'ctc_vocab_size': ctc_output.shape[-1],
        'ed_vocab_size': ed_tokenizer.vocab_size
    }

    return comparison


# Example usage functions
def example_convert_existing_setup():
    """
    Example of how to convert existing CTC setup to Encoder-Decoder.
    """
    # Assuming you have existing CTC setup
    # alphabet = "your_existing_alphabet_string"
    # ctc_converter = CTCLabelConverter(alphabet)

    # Convert to encoder-decoder
    # ed_tokenizer = create_encoder_decoder_tokenizer_from_ctc(ctc_converter)

    # print(f"CTC vocab size: {len(ctc_converter.character)}")
    # print(f"ED vocab size: {ed_tokenizer.vocab_size}")
    # print(f"ED special tokens: SOS={ed_tokenizer.sos_token_id}, EOS={ed_tokenizer.eos_token_id}")

    pass


def example_create_new_tokenizer():
    """
    Example of creating new tokenizer from dataset.
    """
    # Create tokenizer for specific dataset
    # tokenizer, alphabet = create_dataset_specific_tokenizer(
    #     dataset_name='IAM',
    #     train_data_list='./data/iam/train.ln',
    #     data_path='./data/iam/lines/'
    # )

    # print(f"Alphabet: {alphabet}")
    # print(f"Vocab size: {tokenizer.vocab_size}")

    pass
