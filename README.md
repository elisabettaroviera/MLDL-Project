# Semantic Segmentation: Domain Adaptation Project

This repository contains code for the semantic segmentation project of the MLDL course at Politecnico di Torino.
The project has a focus on domain adaptation from synthetic (GTA5) to real-world (Cityscapes) datasets, but it revolves around testing the performance of different solutions. It supports multiple architectures (DeepLabV2, BiSeNet, PIDNet) and advanced training strategies including adversarial learning.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Setup & Installation](#setup--installation)
- [Dataset Preparation](#dataset-preparation)
- [Training & Evaluation](#training--evaluation)
- [Domain Adaptation](#domain-adaptation)
- [WANDB Logging](#wandb-logging)
- [Notebook Usage](#notebook-usage)
- [Experiments](#experiments)
- [Final Results](#final-results)
- [References](#references)
- [Authors](#authors)

---

## Project Overview

- **Goal:** Evaluate the performance of different approaches in the semantic segmentation field.
- **Models:** DeepLabV2, BiSeNet, PIDNet.
- **Techniques:** Standard training, domain adaptation, adversarial training, advanced augmentations.
- **Logging:** [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking and model checkpointing.

---

## Setup & Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/elisabettaroviera/MLDL-Project.git
    cd MLDL-Project
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset Preparation

- **Cityscapes** and **GTA5** datasets are required.
- Download scripts are provided:
    ```bash
    python utils/download_Cityscapes.py
    python utils/download_GTA5.py
    ```
- Place datasets in the `./datasets/` directory or update paths in scripts as needed.

---

## Training & Evaluation

### Standard Training

- **DeepLabV2 or BiSeNet on Cityscapes:**
    ```bash
    python main2ab.py
    ```
    - Model selection via `MODEL` environment variable (`DeepLabV2` or `BiSeNet`) only when relevant.
    - Training and validation on Cityscapes.
    - WANDB used for logging and checkpointing.

### PIDNet Experiments

- **PIDNet on Cityscapes or GTA5:**
    ```bash
    python main5.py --mode main1   # Cityscapes to Cityscapes
    python main5.py --mode main2   # GTA5 to Cityscapes (no aug)
    python main5.py --mode main3   # GTA5 to Cityscapes (aug_1)
    python main5.py --mode main4   # GTA5 to Cityscapes (aug_2)
    ```

---

## Domain Adaptation

### Domain Shift & Adaptation

- **Domain shift experiments (GTA5 → Cityscapes):**
    ```bash
    python main3ab.py
    ```
    - Supports random subset selection, augmentation, and model checkpointing.

- **Adversarial Domain Adaptation:**
    ```bash
    python main4.py
    ```
    - Trains with adversarial loss using discriminators.
    - Supports multiple adversarial strategies (e.g., hinge ramp-up, BCE).

---

## WANDB Logging 

- Set your WANDB API key as an environment variable (follow your cli guidelines on how to do it):
    
```bash
    export WANDB_API_KEY=your_api_key_here
```
- Each epoch is logged as a separate run; models are checkpointed as artifacts and can be resumed.
- It might be necessary to force the API key with the login method inside the python script (like in [main4](main4.py)).
---

## Notebook Usage

- See [TORUN_Notebook.ipynb](TORUN_Notebook.ipynb) for a step-by-step guide:
    - Cloning, setup, dataset download, WANDB setup, and running experiments.
    - Designed for Colab and local runs.

---

## Experiments

During the course of the project many experiments have been performed.
To take a look to all the source code relative to those experiments please surf the different branches of [this repository](https://github.com/elisabettaroviera/MLDL-Project).

---

## Final Results
Finally, after collecting all the data, we made our findings available in the [final report](Semantic%20Segmentation%20-%20A%20Comparative%20Study%20on%20DeepLabV2,%20BiSeNet,%20PIDNet.pdf).
---

## References

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)
- [BiSeNet Paper](https://arxiv.org/abs/1808.00897)
- [DeepLabV2 Paper](https://arxiv.org/abs/1606.00915)
- [PIDNet Paper](https://arxiv.org/abs/2206.02066)

---

## Authors

- Aurona Gashi (`s322791`)
- Lucia Ghezzi (`s325951`)
- Giacomo Maino (`s338682`)
- Elisabetta Roviera (`s328422`)

---

## Notes

- For more details on dataset structure and metrics, see the [read_me](read_me) folder.
- For custom experiments, modify the scripts in the root directory.
- For troubleshooting, see comments in the code and the notebook.

---