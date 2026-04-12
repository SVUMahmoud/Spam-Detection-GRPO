# Binary Spam Detection Using GRPO and LoRA on Qwen 2.5-3B-Instruct

This repository contains the training and evaluation code for the paper:

> **Binary Spam Detection Using GRPO and LoRA on Qwen 2.5-3B-Instruct**  
> Mahmoud E. Ali, Anjali Diwan, Rajendrasinh Jadeja  
> *Arabian Journal for Science and Engineering (AJSE)*

## Overview

This work fine-tunes the [Qwen 2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) generative language model for binary email classification (spam vs. ham) using **Group Relative Policy Optimization (GRPO)** combined with **Low-Rank Adaptation (LoRA)**. The GRPO framework uses rule-based reward functions (correctness and length) to train the model to produce concise, single-word classification outputs.

## Repository Structure

```
├── README.md
├── Qwen_Detection.ipynb     # Full training and evaluation pipeline (Jupyter Notebook)
├── requirements.txt         # Python dependencies with pinned versions
```

## Reproducibility Note

The results reported in the paper were produced using the following software environment on **Google Colab (Tesla T4 GPU)** in **March 2025**:

| Package | Version Used |
|---|---|
| Unsloth | 2025.3.19 |
| TRL | 0.15.2 |
| Transformers | 4.50.0 |
| vLLM | 0.8.2 |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| XFormers | 0.0.29.post3 |
| CUDA Toolkit | 12.4 |

> **Important:** The TRL library's `GRPOTrainer` underwent significant changes between v0.15 and later versions — including reward scaling fixes ([PR #3992](https://github.com/huggingface/trl/pull/3992)), GRPO sampling logic fixes ([PR #3725](https://github.com/huggingface/trl/pull/3725)), a default `beta` change to 0.0 ([PR #3516](https://github.com/huggingface/trl/pull/3516)), and advantage normalization updates. Additionally, newer versions of Unsloth (≥2026.4) automatically override `gpu_memory_utilization` to 0.8 via Standby Mode, whereas the paper's experiments used 0.5 as explicitly configured. Using newer library versions will produce different numerical results. We pin exact versions in `requirements.txt` to ensure reproducibility of the reported figures.

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
| Quantization | 4-bit (BitsAndBytes, nf4, double quantization) |
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
| vLLM Engine | V0 (compute capability < 8.0 fallback) |

## Datasets

The paper evaluates on four publicly available datasets. The notebook demonstrates the pipeline on the **SpamDetection** dataset; the same code can be adapted to the other datasets by modifying the data loading cell.

| Dataset | Source | Total Samples |
|---|---|---|
| SpamClassify | [Hugging Face](https://huggingface.co/datasets/ucirvine/sms_spam) | 5,572 |
| SpamDetection | [Hugging Face](https://huggingface.co/datasets/Deysi/spam-detection-dataset) | 10,900 |
| LingSpam | [Hugging Face](https://huggingface.co/datasets/SetFit/LingSpam) | 2,893 |
| SpamAssassin | [Hugging Face](https://huggingface.co/datasets/talby/spamassassin) | 10,749 |

## Requirements

- Python 3.10+
- NVIDIA GPU with ≥16 GB VRAM (tested on Tesla T4)
- Google Colab (T4 GPU) is sufficient

### Installation

To reproduce the exact results reported in the paper, install the pinned versions:

```bash
pip install --no-deps unsloth vllm
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft "trl==0.15.2" triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
```

See `requirements.txt` for the complete dependency list.

## Usage

### Training

Open `Qwen_Detection.ipynb` in Google Colab (with T4 GPU runtime) and run all cells sequentially. The notebook will:

1. Load and quantize the Qwen 2.5-3B-Instruct model with LoRA
2. Prepare the dataset in the GRPO-compatible format
3. Define correctness and length reward functions
4. Train with `GRPOTrainer` from the TRL library (300 steps)
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

| Component | Specification |
|---|---|
| GPU | NVIDIA Tesla T4 (16 GB VRAM, Compute Capability 7.5) |
| vLLM Engine | V0 (automatic fallback for CC < 8.0) |
| Attention Backend | XFormers (FlashAttention-2 not supported on T4) |

## Citation

If you find this work useful, please cite:

```bibtex
@article{ali2025spam_grpo,
  title={Binary Spam Detection Using GRPO and LoRA on Qwen 2.5-3B-Instruct},
  author={Ali, Mahmoud E. and Diwan, Anjali and Jadeja, Rajendrasinh},
  journal={Arabian Journal for Science and Engineering},
  year={2025}
}
```

## License

This project is released under the [MIT License](LICENSE).
