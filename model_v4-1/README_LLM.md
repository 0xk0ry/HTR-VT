# HTR-VT with RoBERTa Large Language Model Text Correction

This directory contains an enhanced version of HTR-VT that integrates RoBERTa Large language model for post-processing text correction of CTC decoder outputs.

## Overview

The system combines:
1. **HTR-VT Model**: Vision Transformer-based handwritten text recognition
2. **CTC Decoder**: Connectionist Temporal Classification for sequence decoding
3. **RoBERTa Large**: Pre-trained language model for text correction and refinement

## Key Features

- **Iterative Text Correction**: Uses masked language modeling to iteratively improve recognition results
- **Confidence-based Filtering**: Only applies corrections above a configurable confidence threshold
- **Comprehensive Evaluation**: Compares original CTC outputs with LLM-corrected results
- **Detailed Logging**: Provides sample corrections and improvement metrics

## Installation

### Requirements

Install the additional dependencies for LLM functionality:

```bash
pip install transformers>=4.21.0 torch>=1.12.0 editdistance numpy
```

Or install from the requirements file:

```bash
pip install -r requirements_llm.txt
```

## Usage

### 1. Full Test with LLM Correction

Run the complete test with RoBERTa text correction:

**Linux/Mac:**
```bash
cd model_v3
python test_with_llm.py IAM \
    --resume_checkpoint="../best_CER.pth" \
    --train-data-list="../data/iam/train.ln" \
    --val-data-list="../data/iam/val.ln" \
    --test-data-list="../data/iam/test.ln" \
    --data-path="../data/iam/lines/" \
    --nb-cls=79 \
    --val-bs=8 \
    --img-size 512 64 \
    --exp-name="IAM_HTR_ORIGAMI_NET_WITH_LLM" \
    --roberta-confidence-threshold=0.5 \
    --max-correction-iterations=3
```

**Windows PowerShell:**
```powershell
.\run\test_llm_iam.ps1
```

### 2. Quick Single Image Inference

For quick testing on individual images:

```bash
cd model_v3
python quick_inference_llm.py \
    --image_path "path/to/your/image.jpg" \
    --checkpoint "../best_CER.pth" \
    --roberta_model "roberta-large" \
    --confidence_threshold 0.5
```

To disable LLM correction (CTC only):
```bash
python quick_inference_llm.py \
    --image_path "path/to/your/image.jpg" \
    --checkpoint "../best_CER.pth" \
    --disable_correction
```

## Configuration Options

### LLM-Specific Arguments

- `--roberta-model-name`: RoBERTa model variant (default: "roberta-large")
- `--roberta-confidence-threshold`: Minimum confidence for applying corrections (default: 0.5)
- `--enable-llm-correction`: Enable LLM-based text correction
- `--max-correction-iterations`: Maximum iterative correction passes (default: 3)

### Supported RoBERTa Models

- `roberta-base`: Faster, lower memory usage
- `roberta-large`: Better performance, higher accuracy (recommended)
- Other RoBERTa variants from Hugging Face

## Output Files

The test generates several output files in the experiment directory:

1. **`llm_correction_results.json`**: Comprehensive metrics and results
2. **`sample_corrections.txt`**: Sample text corrections for manual inspection
3. **`run.log`**: Detailed execution log with metrics

### Sample Output

```
==========================================
ORIGINAL CTC RESULTS:
Test loss: 0.123   CER: 0.0456   WER: 0.1234
==========================================
ROBERTA-CORRECTED RESULTS:
Test loss: 0.123   CER: 0.0398   WER: 0.1089
==========================================
IMPROVEMENT:
CER improvement: 12.72%
WER improvement: 11.75%
==========================================
```

## How It Works

### 1. CTC Decoding
- Standard CTC beam search decoding produces initial text hypothesis
- May contain character-level errors due to visual ambiguities

### 2. RoBERTa Text Correction
- **Masked Language Modeling**: Systematically masks each word and predicts replacements
- **Confidence Filtering**: Only applies corrections above threshold confidence
- **Iterative Refinement**: Multiple passes to improve text quality
- **Perplexity Scoring**: Uses language model perplexity to select best corrections

### 3. Evaluation
- Compares original CTC outputs with corrected text
- Calculates Character Error Rate (CER) and Word Error Rate (WER)
- Provides detailed improvement metrics

## Performance Considerations

### Memory Usage
- RoBERTa Large requires significant GPU memory (~3-4GB)
- Consider using `roberta-base` for memory-constrained environments
- Batch processing is optimized for efficiency

### Speed
- Text correction adds computational overhead
- Typical processing: 2-5 seconds per image on modern GPU
- Can be disabled for speed-critical applications

### Accuracy Trade-offs
- Higher confidence thresholds: Fewer but more accurate corrections
- Lower confidence thresholds: More corrections but potential over-correction
- Optimal threshold depends on your dataset and use case

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size (`--val-bs`)
   - Use `roberta-base` instead of `roberta-large`
   - Enable CPU fallback

2. **Slow Performance**
   - Reduce `--max-correction-iterations`
   - Increase `--roberta-confidence-threshold`
   - Use smaller RoBERTa model

3. **Poor Correction Quality**
   - Adjust `--roberta-confidence-threshold`
   - Verify input text quality
   - Check character dictionary compatibility

### Debug Mode

Enable detailed logging:
```bash
python test_with_llm.py --debug IAM [other args...]
```

## Citation

If you use this LLM-enhanced version in your research, please cite both the original HTR-VT paper and acknowledge the RoBERTa integration:

```bibtex
@article{htr_vt_llm,
  title={HTR-VT with RoBERTa Language Model Enhancement},
  note={Enhanced version with pre-trained language model text correction},
  year={2024}
}
```

## License

This enhanced version maintains the same license as the original HTR-VT project.
