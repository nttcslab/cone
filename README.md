# Deep Quantigraphic Image Enhancement via Comparametric Equations

![](https://img.shields.io/badge/python-3.7-blue.svg)

Official PyTorch Implementation of [Deep Quantigraphic Image Enhancement via Comparametric Equations]() (ICASSP2023)

```BibTeX
@inproceedings{cone,
  author    = {Xiaomeng Wu and
               Yongqing Sun and
               Akisato Kimura},
  title     = {Deep Quantigraphic Image Enhancement via Comparametric Equations},
  booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  pages     = {xxxx--xxxx},
  year      = {2023},
}
```

------

## Requirements

- Python 3.7
- PyTorch 1.8.0
- scikit-image (only for PSNR and SSIM computations)
- mmcv 1.6.0 (only for computing parameter counts and flops)

For example, the following commands create an execution environment named "cone" for this PyTorch implementation.

```
conda create --name cone python=3.7
conda activate cone
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install scikit-image
pip install mmcv==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html
```

## Data Preparation

Download the `mit` and `lsrw` folders (about 900M total) from [here](https://github.com/xiaomengwupx/cone-data) and store them in the `data` folder. These two benchmark datasets have been reorganized by [Ma et al](https://github.com/vis-opt-group/SCI). In this reorganized configuration, both datasets contain 500 training images. The number of test images is 100 and 50, respectively.


