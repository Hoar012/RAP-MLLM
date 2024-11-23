# RAP-MLLM: Retrieval-Augmented Personalization for Multimodal Large Language Model

### [Paper](https://arxiv.org/abs/2410.13360) | [Project Page](https://hoar012.github.io/RAP-Project/) | [Model (Coming soon)](https://github.com/Hoar012/Rap-MLLM) | [Data (Coming soon)](https://github.com/Hoar012/Rap-MLLM)

## News
- **2024.11.24** Release code.
- **2024.10.17** Release paper.

## Personalize Your Multimodal Large Language Model via Retrieval Augmented Generation.

![RAP Framework](./images/teaser.png)

Visit our [Project Page](https://hoar012.github.io/RAP-Project/) for more demostrations.

## Contents

- [Install](#install)

### Install

1. Clone the repo into a local folder.

```bash
git clone https://github.com/Hoar012/RAP-MLLM.git

cd RAP-MLLM
```

2. Install packages.

```bash
conda create -n rap python=3.10 -y
conda activate rap
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

pip install -r requirements.txt
```

## BibTeX

```
@misc{hao2024rememberretrievegenerateunderstanding,
        title={Remember, Retrieve and Generate: Understanding Infinite Visual Concepts as Your Personalized Assistant}, 
        author={Haoran Hao and Jiaming Han and Changsheng Li and Yu-Feng Li and Xiangyu Yue},
        year={2024},
        eprint={2410.13360},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2410.13360}, 
  }
```