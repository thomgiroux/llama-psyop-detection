# Llama-3.1-8B Fine-Tuned for PSYOP Detection

A fine-tuned version of Meta's Llama-3.1-8B optimized for detecting and analyzing psychological coercion and PSYOP techniques in text.

## Project Overview

This project demonstrates end-to-end machine learning competency through:
- **Dataset curation**: Labeled 12.7k examples from YouTube interview transcripts
- **Model fine-tuning**: LoRA-based adaptation of Llama-3.1-8B
- **Professional deployment**: Published to HuggingFace Hub

## Key Results

| Metric | Value |
|--------|-------|
| Training Loss (final) | 1.071 |
| Validation Loss (final) | 1.075 |
| Training Time | ~35 minutes |
| Hardware | Google Colab A100 GPU |
| Fine-tuning Method | LoRA (r=8, alpha=16) |

## Files & Links

- **Fine-tuned Model**: [LeTG/llama-3p1-8B-psyop-analysis](https://huggingface.co/LeTG/llama-3p1-8B-psyop-analysis)
- **Dataset**: [LeTG/psychological-coercion-identification](https://huggingface.co/datasets/LeTG/psychological-coercion-identification)
- **Training Notebook**: [Colab Notebook](https://colab.research.google.com/drive/1wJkUQ4-vqNk1yHmatW9yYQuk2paprFxS?usp=sharing)

## Methodology

### Data

The training dataset consists of 12,655 labeled examples extracted from YouTube interview transcripts:
- **Training set**: 10,137 examples (80%)
- **Validation set**: 1,254 examples (20%)
- **Labels**: PSYOP presence (binary), confidence score, techniques, target audience, sentiment

### Model Architecture

**Base Model**: Meta's Llama-3.1-8B (8 billion parameters)
**Fine-tuning Method**: Low-Rank Adaptation (LoRA)
- Rank (r): 8
- Alpha (α): 16
- Dropout: 0.05
- Target modules: q_proj, v_proj

**Rationale**: LoRA enables efficient fine-tuning of large models by training only small adapter matrices (~0.04% of parameters), reducing memory requirements and training time while maintaining model quality.

### Training Configuration
```
Epochs: 2
Batch Size: 4 (per device)
Learning Rate: 2e-4
Gradient Accumulation Steps: 4
Max Sequence Length: 256 tokens
Optimizer: AdamW
```

## Results

The model successfully converges over 2 epochs:
- Training loss decreases from 1.118 → 1.071
- Validation loss stabilizes around 1.075
- No signs of overfitting (validation loss tracks training loss)

## Limitations & Future Work

### Current Limitations
- Trained exclusively on YouTube interview transcripts; generalization to other domains unknown
- Limited to English-language content
- Validation metrics focus on loss; task-specific accuracy metrics would strengthen evaluation

### Future Improvements
- Evaluate on held-out test set with task-specific metrics (F1, precision, recall)
- Expand dataset to 50-100k examples for improved robustness
- Quantization for efficient deployment
- Comparison with other fine-tuning approaches (full fine-tuning, other adapters)

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LeTG/llama-3p1-8B-psyop-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Use the model for inference
prompt = "Analyze this text for psychological coercion..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

## Technical Stack

- **ML Framework**: PyTorch + Hugging Face Transformers
- **Fine-tuning Framework**: PEFT (Parameter-Efficient Fine-Tuning)
- **Compute**: Google Colab (A100 GPU)
- **Data Processing**: Hugging Face Datasets
- **Model Hub**: HuggingFace Model Hub

## What This Demonstrates

This project showcases:
1. **Data curation**: Building and labeling a domain-specific dataset
2. **MLOps**: Scaling data pipeline from manual to automated labeling with Claude API
3. **Model optimization**: Efficient fine-tuning using LoRA
4. **Responsible AI**: Documentation of limitations and ethical considerations
5. **Professional communication**: Clear documentation for reproducibility

## About

Created as a portfolio project to demonstrate practical ML engineering skills for AI research roles.

---

*For questions or feedback, open an issue or reach out.*
