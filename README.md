# [Hybrid Functional Maps for Crease-Aware Non-Isometric Shape Matching [ECCV 2026]](https://luofeifan77.github.io/publications/)
 [![PDF](https://img.shields.io/badge/PDF-Download-blue)](https://arxiv.org/pdf/2606.26557)
<!--[![ArXiv](https://img.shields.io/badge/arXiv-2312.03678-b31b1b.svg)](https://arxiv.org/abs/2312.03678)-->

![img](figures/ecc26_teaser.png)

## Installation
Our code relies on PyTorch, along with several other common libraries. We recommend to use our provided conda environment file for compatibility:
```bash 
# create new virtual environment
conda env create --name coarse2fine -f environment.yml

conda activate coarse2fine
```
In addition, this code uses python bindings for an implementation of the Discrete Shell Energy. 

Please follow the installation instructions from: [Thin shell energy](https://gitlab.com/numod/shell-energy)

## Dataset
For training and testing datasets used in this paper, please refer to the [ULRSSM repository](https://github.com/dongliangcao/Unsupervised-Learning-of-Robust-Spectral-Shape-Matching/) from Dongliang Cao et al. Please follow the instructions there to download the necessary datasets and place them under `../data/`: 
```Shell
├── data
    ├── FAUST_r
    ├── FAUST_a
    ├── SCAPE_r
    ├── SCAPE_a
    ├── SHREC19_r
    ├── TOPKIDS
    ├── SMAL_r
    ├── DT4D_r
```
We thank the original dataset providers for their contributions to the shape analysis community, and that all credits should go to the the respective authors and contributors.

## Data preparation
For data preprocessing, we provide *[preprocess.py](preprocess.py)* to compute all things we need.
Here is an example for SMAL_r.
```python
python preprocess.py --data_root ../data/SMAL_r/ --no_normalize --n_eig 200
```

## Train
To train a specific model on a specified dataset.
```python
python train.py --opt options/hybrid_ulrssm/train/smal.yaml
```
You can visualize the training process in tensorboard or via wandb.
```bash
tensorboard --logdir experiments/
```

## Test
To test a specific model on a specified dataset.
```python
python test.py --opt options/hybrid_ulrssm/test/smal.yaml
```
The qualitative and quantitative results will be saved in [results](results) folder.

<!-- ## Texture Transfer
An example of texture transfer is provided in *[texture_transfer.py](texture_transfer.py)*
```python
python texture_transfer.py
``` -->

## Visualization
Make sure to install the latest [polyscope](https://github.com/nmwsharp/polyscope) to allow headless rendering.
```
pip uninstall polyscope
pip install git+https://github.com/nmwsharp/polyscope-py.git
```
To visualize the final results.
```python
python visualize.py --opt options/hybrid_ulrssm/test/smal.yaml
```
The visualized images will be saved in [results](results) folder.
## Pretrained models
You can find all pre-trained models in [checkpoints](checkpoints) for reproducibility.

## Acknowledgement
The framework implementation is adapted from [Hybrid Functional Maps for Crease-Aware Non-Isometric Shape Matching](https://github.com/xieyizheng/hybridfmaps/).

The implementation of Elastic Basis is adapted from [An Elastic Basis for Spectral Shape Correspondence](https://github.com/flrneha/ElasticBasisForSpectralMatching/).

The implementation of DiffusionNet is based on [the official implementation](https://github.com/nmwsharp/diffusion-net).

The implementation of map refinement modules follow [DeepFAFM](https://github.com/LuoFeifan77/DeepFAFM).



We thank the original authors for their contributions to this code base.

<!-- : [Nickolas Sharp](https://github.com/nmwsharp/), [Florine Hartwig](https://github.com/flrneha) and [Dongliang Cao](https://github.com/dongliangcao), -->

## Attribution
Please cite our paper when using the code. You can use the following bibtex
```
@article{luo2026coarse,
  title={Coarse-to-Fine: A Hybrid Self-Supervised Method for Non-rigid 3D Shape Matching},
  author={Luo, Feifan and Li, Ting and Li, Zhao and Chen, Hongyang},
  journal={arXiv preprint arXiv:2606.26557},
  year={2026}
}

```
