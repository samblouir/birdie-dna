# Birdie-DNA

This repository is based on Birdie's training code and is being rapidly updated.
It currently supports a Transformer+++ model and supporting scripts to train it on DNA sequences.

The code is designed to be both convenient and hackable. It supports tasks like:

- Automated downloading and parsing of genomic FASTA files  
- Tokenizing DNA strings at the byte-level (ByT5 style)  
- Running training and evaluation steps via Hugging Face Accelerate  
- Applying custom model layers and modules, like rotary embeddings, RMSNorm, GQA, softcapping, and more  

**Table of Contents**  
1. [Project Overview](#1-project-overview)  
2. [Installation & Requirements](#2-installation--requirements)  
3. [Repository Structure](#3-repository-structure)  
4. [Usage](#4-usage)  
   1. [Quick Start Training](#41-quick-start-training)  
   2. [Configuration](#42-configuration)  
   3. [Dataset Handling](#43-dataset-handling)  
5. [Key Scripts & Modules](#5-key-scripts--modules)  
   1. [Configs](#51-configs-folder)  
   2. [Data Utilities](#52-data_utils-folder)  
   3. [Modeling](#53-modeling-folder)  
   4. [Training & Evaluation Scripts](#54-training--evaluation-scripts)  
6. [Notes on Code Quality / Potential Issues](#6-notes-on-code-quality--potential-issues)  
7. [License](#7-license)  

---

## 1) Project Overview

 Currently, this trains a Transformer language model on DNA sequences (currently genomic data from T2T)

- **Tokenizer**: A ByT5-like byte-level tokenizer for DNA characters, and is easily replaceable.  
- **Data Loader**: Automatic download and extraction of T2T.
- **Model**: A “BaseModel” that uses standard Transformer blocks with optional features (e.g., rotary embeddings, GQA).  
- **Training** and **Evaluation**: Uses [Hugging Face Accelerate](https://github.com/huggingface/accelerate) to handle multi-machine and multi-GPU setups.

---

## 2) Installation & Requirements

1. **Clone the repo or copy files**:

   ```bash
   git clone https://github.com/samblouir/birdie-dna.git
   cd birdie-dna
   ```

2. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   The key libraries are:
   - **PyTorch** (version 2.7.0 or higher, GPU recommended)
   - **accelerate** + **tqdm** + **biopython** + **einops**
   - **numpy**  
   Make sure you also have a recent Python 3.8+ environment.

3. **(Optional) Setup a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 3) Repository Structure

Below is the tree of main files and directories:

```
configs/
  default.py
  mini.py
data_utils/
  dataloader.py
modeling/
  basemodel.py
  prepare_optimizer.py
  rotary.py
  softcap.py
  tokenizer.py
.gitignore
config.py
configure_accelerator.py
evaluate_model.py
readme.md
requirements.txt
train.py
utils.py
```

| Path / File                     | Description                                                                      |
|--------------------------------|----------------------------------------------------------------------------------|
| **configs/**                   | Contains Python files that return dicts of hyperparameters (default, mini, etc.).|
| **data_utils/dataloader.py**   | Code that handles downloading, unpacking, parsing FASTA, splitting train/val/test, and creating batch generators.|
| **modeling/basemodel.py**                  | Code related to the model architecture, rotary embeddings, attention, etc.       |
| **train.py**                   | Main training script (run with `accelerate launch`).                             |
| **evaluate_model.py**          | Evaluation script that loads a config + model checkpoints to compute test metrics (supports 'accelerate launch' for speedier evals..|
| **configure_accelerator.py**   | Utility to initialize and configure a Hugging Face `Accelerator`.                |
| **utils.py**                   | General-purpose helpers (debug printing, hashing, logging, etc.).               |
| **requirements.txt**           | Python package dependencies.                                                    |
| **config.py**                  | Global flags (debug prints, some environment variables).                              |
| **readme.md**                  | This README                                          |

---

## 4) Usage

### 4.1 Quick Start Training

To train using a configuration (say `configs/mini.py`), do:

```bash
accelerate launch train.py --config=mini
```
This will:

1. Load hyperparameters from `configs/mini.py`.  
2. Automatically download the T2T reference genome (or other specified sources) if needed.  
3. Parse data into train/validation/test splits.  
4. Build the Transformer (`BaseModel` in `modeling/basemodel.py`).  
5. Run training steps, periodically logging training loss and evaluating on the validation set.  
6. Save checkpoints under `saves/checkpoints/checkpoint_<step>`.

#### Common Flags

- `--config=<name>` picks which config file in `configs/` to load.
- You can fine-tune or skip certain steps via your code or config.

### 4.2 Quick Start Evaluation

After training (or while checkpoints are being saved), you can evaluate the model on the test split using **`evaluate_model.py`**. For example:

```bash
accelerate launch evaluate_model.py --config=mini
```

It will:

1. Load the same config (`mini.py`) so it knows hyperparameters like sequence length, etc.  
2. Search `saves/checkpoints` for existing checkpoints (e.g., `checkpoint_512`, `checkpoint_1024`, etc.).  
3. For each checkpoint found, restore model state and compute average test loss over a small number of test batches.  
4. Log the results in `test_losses.txt` or console output.  

If you only want to evaluate the final checkpoint, you can modify `evaluate_model.py` or remove older checkpoints manually.

### 4.3 Configuration

Configs are simple Python modules returning a dictionary. For example, [`configs/default.py`](configs/default.py) has fields like:

```python
{
  "config_name": "default",
  "sequence_length": 1024,
  "batch_size": 32,
  "num_steps": 32768,
  "eval_interval": 512,
  ...
}
```

You can create a new config (e.g., `configs/big.py`) and override any hyperparameters for your larger model. Then run `accelerate launch train.py --config=big`.

### 4.4 Dataset Handling

- The script `data_utils/dataloader.py` is a data loader that should be split up into seperate files.
Currently, it can:
  1. **Download** an archive from NCBI (or a custom URL).  
  2. **Extract** the file if it’s `.zip` or `.tar.gz`.  
  3. **Parse** the resulting FASTA using BioPython.  
  4. **Split** the data into train, validation, test according to user-specified percentages.  
  5. **Batch** the sequences for causal language modeling (CLM).  

By default, it downloads the [T2T genome](https://www.ncbi.nlm.nih.gov/assembly/GCF_009914755.1/) if no custom files are specified.

---

## 5) Key Scripts & Modules

### 5.1 `configs/` folder

- **`default.py`**: The baseline hyperparameter dictionary (e.g., 16 layers, hidden size=512, etc.).  
- **`mini.py`**: Inherits from `default.py` but overrides a few values to produce a smaller, faster demo.  

### 5.2 `data_utils/` folder

- **`dataloader.py`**:  
  - **`download_datasets()`**: Download + optionally extract archives.  
  - **`load_dataset()`**: The main function for returning train/val/test splits.  
  - **`create_clm_batch_generator()`**: Produces tokenized mini-batches for next-token prediction.  

### 5.3 `modeling/` folder

Inside `modeling/`:

1. **`basemodel.py`**  
   - **`BaseModel`**: The main Transformer model that combines embeddings, multi-head attention, feedforward blocks, and a final RMSNorm + linear head. Also handles optional features like GQA, rotary embeddings, and fused cross-entropy.

2. **`rotary.py`**  
   - Functions for computing and applying rotary positional embeddings.

3. **`softcap.py`**  
   - Implements a “tanh softcap” for attention logits. Currently it provides a function to generate a tanh-based capping function.

4. **`prepare_optimizer.py`**  
   - Creates an AdamW optimizer with separate param groups and a warmup + cosine decay scheduler.

5. **`tokenizer.py`**  
   - A ByT5-based DNA tokenizer that can handle any text by bytes.  
   - If you want a specialized A/C/G/T-only or any other custom tokenizer, you can adapt it here.

### 5.4 Training & Evaluation Scripts

1. **`train.py`**  
   - Expects a `--config=<name>` argument.  
   - Loads the config, sets up an accelerator, initializes the model + optimizer, sets up the dataset, and runs the training loop.  
   - Saves model checkpoints in `./saves/checkpoints/checkpoint_<step>`.

2. **`evaluate_model.py`**  
   - Similarly loads a config + accelerator.  
   - Finds all checkpoints in `saves/<config_name>/checkpoints`.  
   - Evaluates on the test split or any custom dataset.  
   - Writes the test loss to `test_losses.txt` or logs accordingly.

3. **`configure_accelerator.py`**  
   - A helper that sets up a [Hugging Face Accelerator](https://github.com/huggingface/accelerate) instance with sensible defaults (mixed-precision, project directories, etc.).

4. **`utils.py`**  
   - Contains several convenience methods for hashing, logging, moving batches to devices, etc.  
   - Also includes a `debug_print()` function that depends on a global flag set in `config.py`.

---

## 6) Citing and License

If you use the code, please cite [Birdie's paper](https://arxiv.org/abs/2411.01030).

```
Apache 2.0 License
```
---