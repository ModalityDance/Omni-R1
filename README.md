<a name="readme-top"></a>

<div align="center">
  <h1 align="center">Omni-R1: Towards the Unified Generative Paradigm for Multimodal Reasoning</h1>
</div>

<div align="center">
  <!-- Paper Link -->
  <a href="">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="Paper">
  </a>

  <!-- HuggingFace Models -->
  <a href="https://huggingface.co/ModalityDance/Omni-R1">
    <img src="https://img.shields.io/badge/HuggingFace-Omni--R1-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="Omni-R1">
  </a>
    
  <a href="https://huggingface.co/ModalityDance/Omni-R1-Zero">
    <img src="https://img.shields.io/badge/HuggingFace-Omni--R1--Zero-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="Omni-R1-Zero">
  </a>

  <!-- Omni-Bench Badge -->
  <a href="https://huggingface.co/datasets/ModalityDance/Omni-Bench">
    <img src="https://img.shields.io/badge/Omni--Bench-Available-4c1?style=for-the-badge" alt="Omni-Bench">
  </a>
    <img src="./assets/overview.png" alt="vision">
</div>

---
Welcome to **Omni-R1**! 👋 This repository provides implementation code for *"Omni-R1: Towards the Unified Generative Paradigm for Multimodal Reasoning"*.

We instantiate this paradigm with Omni-R1, a two-stage SFT+RL framework featuring perception alignment loss and perception reward, thereby enabling functional image generation. Additionally, we introduce Omni-R1-Zero, which eliminates the need for multimodal annotations by bootstrapping step-wise visualizations from text-only reasoning data. 

### 🪐 Key Features
> [!IMPORTANT]
> Faster Evaluation & RL Rollouts with vLLM. Our evaluation and RL rollout pipelines(based on verl) are accelerated by vLLM, which can significantly reduce the inference time of large-scale sampling and long rollouts.

🧭 **Two-stage training pipeline**  
PeSFT introduces perception alignment loss during SFT, and PeRPO applies a perception reward during RL to enhance functional image generation.

🌌 **Two training regimes under different supervision**  
Omni-R1 uses multimodal interleaved supervision, while Omni-R1-Zero bootstraps step-wise visualizations from text-only reasoning data.

🧩 **Benchmark**  
Includes Omni-Bench data and a vLLM-based evaluation script that runs inference efficiently and saves predictions in JSONL format.


## 🔥 News

<div style="max-height: 240px; overflow-y: auto;">

- **[2026.01]** Initial release of Omni-R1.

</div>

## Roadmap

✅ Reproducibility essentials for Omni-R1 (core code, datasets, checkpoints)  
✅ Paper link  
✅ Omni-Bench (data + vLLM evaluation script)     
⬜ Fully end-to-end PeRPO training framework  
⬜ The implementation of bootstrapping step-wise visualizations


## 📑 Table of Contents <span id="table-of-contents"></span>
* [🚀 Quick Start](#quick-start)
  * [1. Installation](#installation)
    * [Create environment (For Inference & PeSFT)](#install-env)
    * [PeRPO dependency](#install-perpo)
  * [2. Train](#train)
    * [Data](#train-data)
    * [PeSFT](#train-pesft)
    * [PeRPO](#train-perpo)
  * [3. Inference](#inference)
    * [Run](#inference-run)
  * [4. Omni-Bench](#omni-bench)
    * [Data](#omni-bench-data)
    * [Evaluation](#omni-bench-eval)
* [✨ How It Works](#how-it-works)
* [🗂️ Project Structure](#project-structure)
* [🌱 Acknowledgements](#acknowledgements)
* [📚 Citation](#citation)


## 🚀 Quick Start <span id="quick-start"></span>
### 1. Installation <span id="installation"></span>

#### Create environment (For Inference & PeSFT) <span id="install-env"></span>
```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install ./src/transformers
```

#### PeRPO dependency <span id="install-perpo"></span>
```bash
git clone https://github.com/volcengine/verl && cd verl
# Follow the official install docs:
# https://verl.readthedocs.io/en/latest/start/install.html
```
---

### 2. Train <span id="train"></span>
#### Data <span id="train-data"></span>
- **Omni-R1** supervision: Zebra-CoT  
  https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT
- **Omni-R1-Zero** text-only CoT seeds: M3CoT  
  https://huggingface.co/datasets/LightChen2333/M3CoT
#### PeSFT <span id="train-pesft"></span>
**Minimal DeepSpeed config:**
```bash
DS_JSON='{
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2},
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1
}'
```

**Run**
```bash
export BASE_OR_CKPT=/path/to/base_or_ckpt
export OUT=checkpoints/pesft_run
export JSON_DIR=data/zebra_cot_jsonl

deepspeed --num_gpus 8 src/PeSFT/pesft.py \
  --model_path "$BASE_OR_CKPT" \
  --output_path "$OUT" \
  --json_dir "$JSON_DIR" \
  --deepspeed_config_json "$DS_JSON" \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --mode templated # for Omni-R1-Zero
  # --mode plain # for Omni-R1
```

<details>
<summary>What PeSFT does?</summary>

- Supervised finetuning with cross-entropy + **perception alignment loss** to stabilize / improve functional image generation.

</details>

#### PeRPO <span id="train-perpo"></span>
> [!NOTE]
> The end-to-end PeRPO training recipe is being cleaned up and will be released in a more complete, reproducible form soon.

> [!TIP]
> PeRPO can be reproduced by following **verl**’s **DAPO** recipe.
> In `volcengine/verl`, you can directly follow:
> - `verl/recipe/dapo`
>
> Then, plug in and reuse our reward functions in `src/PeRPO/rewards.py` as the reward module for the DAPO training loop.

**Reward implementation:** `src/PeRPO/rewards.py`


<details>
<summary>What PeRPO does?</summary>

- RL refinement with group-relative optimization using a perception-calibrated reward:
  - **Accuracy**
  - **Format**
  - **Perception**
</details>

---

### 3. Inference <span id="inference"></span>
You can skip training with our pretrained models below:

***Checkpoints***
- Omni-R1: https://huggingface.co/ModalityDance/Omni-R1
- Omni-R1-Zero: https://huggingface.co/ModalityDance/Omni-R1-Zero

#### Run <span id="inference-run"></span>
```bash
export INPUT_JSONL=/path/to/data.jsonl
export OUTDIR=outputs/demo_run
export MODEL=/path/to/ckpt
export PROCESSOR=/path/to/processor_ckpt

python src/Inference/inference.py \
  --input "$INPUT_JSONL" \
  --output-dir "$OUTDIR" \
  --model-path "$MODEL" \
  --processor-path "$PROCESSOR" \
  --append-boi \
  --do-sample \
  --temperature 1.0 \
  --top-p 0.9
```


<details>
<summary>Key args meaning</summary>

- `--input`: JSONL file (or a directory of JSONL files)
- `--output-dir`: where predictions are saved
- `--model-path`: your checkpoint
- `--processor-path`: processor checkpoint path
- `--append-boi`: appends BOI token (if your model expects it)
- `--do-sample`, `--temperature`, `--top-p`: sampling settings

</details>

---

### 4. Omni-Bench <span id="omni-bench"></span>

#### Data <span id="omni-bench-data"></span>
Download the dataset: https://huggingface.co/datasets/ModalityDance/Omni-Bench

**Omni-Bench** contains **800 samples** spanning **4 Uni-Tasks**:
- **Natural-Scene Perception**: V\*
- **Structured-Image**: ArxivQA, ChartQA
- **Diagrammatic Math**: Geometry3k, MathVista
- **Vision-Operational Scenes**: ViC-Bench

#### Evaluation <span id="omni-bench-eval"></span>
```bash
python omni-bench/vllm_eval.py \
  --parquet_path omni-bench/omni-bench.parquet \
  --model_path /path/to/your_model \
  --outfile preds.jsonl \
  --mm_images_per_prompt 5
```

<details>
<summary>What this script does?</summary>

- Loads Omni-Bench parquet
- Runs batched inference with vLLM
- Saves predictions in JSONL format (`preds.jsonl`)

</details>

---

## ✨ How It Works <span id="how-it-works"></span>
Omni-R1 learns to generate interleaved multimodal reasoning trajectories through a two-stage SFT → RL pipeline.

- **Omni-R1:** is trained on annotated interleaved multimodal trajectories.
- **Omni-R1-Zero:** when such annotations are unavailable, bootstraps interleaved trajectories from text-only CoT by visualizing per reasoning step, and then trains with the same pipeline.
- **PeSFT:** performs supervised fine-tuning with cross-entropy plus a perception alignment loss to stabilize the functional image generation.
- **PeRPO:** refines the policy with group-relative RL on unified tasks using a composite reward—Accuracy, Format, and Perception.

A high-level overview is illustrated in the figure below.

<div align="center">
  <figure>
    <img src="./assets/framework.png" alt="Overview" style="max-width: 100%; height: auto;">
    <br>
  </figure>
</div>

## 🗂️ Project Structure <span id="project-structure"></span>

```plaintext
.
├── omni-bench/
│   ├── omni-bench.parquet         # Benchmark dataset (Available in HF)
│   └── vllm_eval.py               # vLLM inference / evaluation
│
└── src/
    ├── Inference/
    │   └── inference.py            # Inference
    │
    ├── PeRPO/
    │   └── rewards.py              # Perception reward utilities
    │
    ├── PeSFT/
    │   ├── perception.py           # Perception module
    │   ├── perception_module.ckpt  # Perception module checkpoint
    │   ├── pesft.py                # PeSFT training
    │   └── trainer.py              # Training utilities
    │
    └── transformers/
```

## 🌱 **Acknowledgements** <span id="acknowledgements"></span>

We would like to thank the contributors, open-source projects, and research communities whose work made **Omni-R1** possible.

<!-- Acknowledgement tags (badges) -->
[![Anole](https://img.shields.io/badge/Model-Anole-blue?style=flat&logo=github)](https://github.com/GAIR-NLP/anole)
[![Anole_Training](https://img.shields.io/badge/Code-thinking--with--generated--images-blue?style=flat&logo=github)](https://github.com/GAIR-NLP/thinking-with-generated-images)
[![Training%20Experience](https://img.shields.io/badge/Training%20Exp-ICLR%20Blog-blue?style=flat&logo=gitbook)](https://iclr-blogposts.github.io/2025/blog/fine-tuning-token-based-large-multimodal-models/)
[![Zebra-CoT](https://img.shields.io/badge/Dataset-Zebra--CoT-blue?style=flat&logo=huggingface)](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)
[![M3CoT](https://img.shields.io/badge/Dataset-M3CoT-blue?style=flat&logo=huggingface)](https://huggingface.co/datasets/LightChen2333/M3CoT)
[![Fine--tuning](https://img.shields.io/badge/Fine--tuning-Transformers-blue?style=flat&logo=github)](https://github.com/huggingface/transformers)
[![verl](https://img.shields.io/badge/RL--Framework-verl-blue?style=flat&logo=github)](https://github.com/volcengine/verl)
[![vllm](https://img.shields.io/badge/Inference-vLLM-blue?style=flat&logo=github)](https://github.com/vllm-project/vllm)


This project is licensed under the **MIT License**. Please refer to the LICENSE file for more details.

## 📚 **Citation** <span id="citation"></span>

If you use **Omni-R1** in your research or applications, please consider citing:

```bibtex
@article{omnir1,
  title        = {Omni-R1: Towards the Unified Generative Paradigm for Multimodal Reasoning},
  author       = {},
  journal      = {arXiv preprint arXiv:{}},
  year         = {}
}
```

---

<div align="center">

<a href="https://github.com/ModalityDance/Omni-R1">
  <img src="https://img.shields.io/badge/⭐ Star%20us%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>

<a href="https://github.com/ModalityDance/Omni-R1/issues">
  <img src="https://img.shields.io/badge/🐞 Report%20Issues-e74c3c?style=for-the-badge&logo=github" />
</a>


</div>
