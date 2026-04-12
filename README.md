# Binary Spam Detection Using GRPO and LoRA on Qwen 2.5-3B-Instruct

This repository contains the training and evaluation code for the paper:

> **Binary Spam Detection Using GRPO and LoRA on Qwen 2.5-3B-Instruct**
> Mahmoud E. Ali, Anjali Diwan, Rajendrasinh Jadeja
> *Arabian Journal for Science and Engineering (AJSE), 2026*

## Overview

This work fine-tunes the [Qwen 2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) generative language model for binary email classification (spam vs. ham) using **Group Relative Policy Optimization (GRPO)** combined with **Low-Rank Adaptation (LoRA)**. The GRPO framework uses rule-based reward functions (correctness and length) to train the model to produce concise, single-word classification outputs.

## Repository Structure

```
├── GRPO_Qwen2.5_3B_instruct_SpamDetection.ipynb   # Training and evaluation notebook (Jupyter Notebook)
├── requirements.txt                               # Pinned library versions
└── README.md                                      
```

## Requirements

- Google Colab (or any environment with an NVIDIA GPU with ≥15 GB VRAM)
- Python 3.11+
- Tested on Tesla T4

### Key Dependencies

The results reported in the paper were produced using the following software environment on **Google Colab (Tesla T4 GPU)** in **March 2025**:

| Package | Version |
|---------|---------|
| PyTorch | 2.6.0+cu124 |
| Unsloth | 2025.3.19 |
| TRL | 0.15.2 |
| Transformers | 4.50.0 |
| vLLM | 0.8.2 |
| Accelerate | 1.5.2 |
| XFormers | 0.0.29.post3 |

All dependencies are pinned in `requirements.txt`. On Colab, use the install cell in the notebook.

## Key Components

### Reward Functions (Rule-Based — No Learned Reward Model)

This framework does **not** use a trained neural reward model. Instead, two deterministic, rule-based reward functions guide the GRPO training:

| Reward Function | Condition | Value |
|---|---|---|
| **Correctness Reward** | Prediction matches ground truth | +10 |
| **Correctness Reward** | Prediction is incorrect | −5 |
| **Length Reward** | Response is exactly 1 word | +5 |
| **Length Reward** | Response has *n* words (*n* > 1) | −*n* |

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base Model | Qwen/Qwen2.5-3B-Instruct (4-bit quantized) |
| LoRA Rank | 64 |
| LoRA Alpha | 64 |
| Learning Rate | 5e-6 |
| Quantization | 4-bit
| LR Scheduler | Cosine |
| Warmup Ratio | 0.1 |
| Batch Size | 8 |
| Optimizer | AdamW 8-bit |
| Adam β₁ / β₂ | 0.9 / 0.99 |
| Weight Decay | 0.1 |
| Max Grad Norm | 0.1 |
| Batch Size | 8 |
| Num Generations (Group Size) | 8 |
| Max Steps | 300 |
| Max Prompt Length | 1,024 |
| Max Completion Length | 50 |
| GPU Memory Utilization | 0.5 |

## Datasets

The paper evaluates on four publicly available datasets. The notebook demonstrates the pipeline on the **SpamDetection** dataset; the same code can be adapted to the other datasets by modifying the data loading cell.

| Dataset | Source | Total Samples |
|---|---|---|
| SpamClassify | [mltrev23/spam-classify](https://huggingface.co/datasets/mltrev23/spam-classify) | 5,572 |
| SpamDetection | [Deysi/spam-detection-dataset](https://huggingface.co/datasets/Deysi/spam-detection-dataset) | 10,900 |
| LingSpam | [mandygu/lingspam-dataset](https://www.kaggle.com/datasets/mandygu/lingspam-dataset) | 2,893 |
| SpamAssassin | [talby/spamassassin](https://huggingface.co/datasets/talby/spamassassin) | 10,749 |

## Usage

### Training

Open `GRPO_Qwen2.5_3B_instruct_SpamDetection.ipynb` in Google Colab (with T4 GPU runtime) and run all cells sequentially. The notebook will:

1. Load and quantize the Qwen 2.5-3B-Instruct model with LoRA
2. Prepare the dataset in the GRPO-compatible format
3. Define correctness and length reward functions
4. Train with `GRPOTrainer` from the TRL library (300 steps)
5. Save the trained LoRA adapter to `grpo_saved_lora/`

The notebook includes saved outputs from our original run for reference.

### Evaluation

After training, the notebook evaluates the model on the test split and reports:

- Accuracy, Precision, Recall, F1-Score
- Single-Word Accuracy (SWA)
- Confusion Matrix

## Pretrained Weights

Pretrained model weights (LoRA adapters) are **not** included in this repository due to file size and redistribution constraints on the base model. However, the full training procedure is reproducible using the publicly available base model and the provided code. Training on a T4 GPU takes approximately 2 hours per dataset.

## Citation

If you find this work useful, please cite:

```bibtex
@article{ali2026spam,
  title={Binary Spam Detection Using GRPO and LoRA on Qwen 2.5-3B-Instruct},
  author={Ali, Mahmoud E. and Diwan, Anjali and Jadeja, Rajendrasinh},
  journal={Arabian Journal for Science and Engineering},
  year={2026}
}
```

## License

This project is released for academic and research purposes.
