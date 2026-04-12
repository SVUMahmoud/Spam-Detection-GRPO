# Binary Spam Detection Using GRPO and LoRA on Qwen 2.5-3B-Instruct

This repository contains the training and evaluation code for the paper:

> **Binary Spam Detection Using GRPO and LoRA on Qwen 2.5-3B-Instruct**  
> Mahmoud E. Ali, Anjali Diwan, Rajendrasinh Jadeja  
> *Arabian Journal for Science and Engineering (AJSE)* — Under Review

## Overview

This work fine-tunes the [Qwen 2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) generative language model for binary email classification (spam vs. ham) using **Group Relative Policy Optimization (GRPO)** combined with **Low-Rank Adaptation (LoRA)**. The GRPO framework uses rule-based reward functions (correctness and length) to train the model to produce concise, single-word classification outputs.

## Repository Structure

```
├── README.md
├── Qwen_Detection.ipynb     # Full training and evaluation pipeline (Jupyter Notebook)
├── requirements.txt         # Python dependencies
```

## Key Components

### Reward Functions (Rule-Based — No Learned Reward Model)

This framework does **not** use a trained neural reward model. Instead, two deterministic, rule-based reward functions guide the GRPO training:

| Reward Function | Condition | Value |
|---|---|---|
| **Correctness Reward** | Prediction matches ground truth | +10 |
| **Correctness Reward** | Prediction is incorrect | −5 |
| **Length Reward** | Response is exactly 1 word | +5 |
| **Length Reward** | Response has *n* words (*n* > 1) | −*n* |

### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Base Model | `Qwen/Qwen2.5-3B-Instruct` |
| LoRA Rank | 64 |
| LoRA Alpha | 64 |
| LoRA Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | 4-bit (BitsAndBytes) |
| Learning Rate | 5e-6 |
| Optimizer | AdamW 8-bit |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.1 |
| Batch Size | 8 |
| Gradient Accumulation Steps | 1 |
| Max Training Steps | 300 |
| Num Generations (Group Size) | 8 |
| Max Prompt Length | 1024 tokens |
| Max Completion Length | 50 tokens |
| GPU Memory Utilization | 0.5 |
| Gradient Checkpointing | Enabled (Unsloth) |

## Datasets

The paper evaluates on four publicly available datasets. The notebook demonstrates the pipeline on the **SpamDetection** dataset; the same code can be adapted to the other datasets by modifying the data loading cell.

| Dataset | Source | Total Samples | Balanced |
|---|---|---|---|
| SpamClassify | [Hugging Face](https://huggingface.co/datasets/ucirvine/sms_spam) | 5,572 | No |
| SpamDetection | [Hugging Face](https://huggingface.co/datasets/Deysi/spam-detection-dataset) | 10,900 | Yes |
| LingSpam | [Hugging Face](https://huggingface.co/datasets/SetFit/LingSpam) | 2,893 | No |
| SpamAssassin | [Hugging Face](https://huggingface.co/datasets/talby/spamassassin) | 10,749 | No |

## Requirements

- Python 3.10+
- NVIDIA GPU with ≥16 GB VRAM (tested on Tesla T4)
- Google Colab (T4 GPU) is sufficient

### Installation

```bash
pip install unsloth vllm
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
```

See `requirements.txt` for the full dependency list.

## Usage

### Training

Open `Qwen_Detection.ipynb` in Google Colab (with T4 GPU runtime) and run all cells sequentially. The notebook will:

1. Load and quantize the Qwen 2.5-3B-Instruct model with LoRA
2. Prepare the dataset in the GRPO-compatible format
3. Define correctness and length reward functions
4. Train with `GRPOTrainer` from the TRL library
5. Save the trained LoRA adapter to `grpo_saved_lora/`

### Evaluation

After training, the notebook evaluates the model on the test split and reports:

- Accuracy, Precision, Recall, F1-Score
- Single-Word Accuracy (SWA)
- Exact Match (EM)
- Confusion Matrix

### Inference

```python
from vllm import SamplingParams

system_prompt = "Classify the following message as 'spam' or 'ham'. Output only one word. \n"

text = tokenizer.apply_chat_template([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Your email message here"},
], tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1024)

output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text
```

## Pretrained Weights

Pretrained model weights (LoRA adapters) are **not** included in this repository due to file size and redistribution constraints on the base model. However, the full training procedure is reproducible using the publicly available base model and the provided code. Training on a T4 GPU takes approximately 1–2 hours per dataset.

## Hardware

All experiments were conducted on Google Colab with the following hardware:
- **GPU:** NVIDIA Tesla T4 (16 GB VRAM)
- **CPU:** 2 cores
- **RAM:** 12.7 GB

## Citation

If you find this work useful, please cite:

```bibtex
@article{ali2026spam_grpo,
  title={Binary Spam Detection Using GRPO and LoRA on Qwen 2.5-3B-Instruct},
  author={Ali, Mahmoud E. and Diwan, Anjali and Jadeja, Rajendrasinh},
  journal={Arabian Journal for Science and Engineering},
  year={2026},
  note={Under Review}
}
```

## License

This project is released under the [MIT License](LICENSE).
