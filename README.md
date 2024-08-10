# VideoCutMix

This is the Official Implementation of the Paper: "VideoCutMix: Temporal Segmentation of Surgical Videos in Scarce Data Scenarios", accepted for publication in MICCAI 2024. 

[](extras/teaser.pdf)

# Data Preparation

Incase you want to extract your own features, please follow the instructions specified. The dataset can be obtained from the following link.

Download the preprocessed dataset from the link.

# Installation

The following repository has been tested in CUDA 11.7, Python=3.7, PyTorch=1.13.1

```
conda create --name=videocutmix python=3.7.16
conda activate videocutmix
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install PyYAML==5.4.1
```

# Getting Started

To train the model just run the following command. 
```
bash scripts/run.sh
```
A results.json file will be created at the end of the process which consists of results for the experiment.

# Citation

If you find this repository useful, please use the following BibTeX entry for citation.
