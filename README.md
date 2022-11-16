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
  year      = {2023}
}
```

------

## Requirements

- Python 3.7
- PyTorch 1.8.0
- scikit-image
- mmcv 1.6.0 (only for computing parameter counts and flops)

For example, the following commands create an execution environment named `cone` for this PyTorch implementation.

```bash
conda create --name cone python=3.7
conda activate cone
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install scikit-image
pip install mmcv==1.6.0
```

## Data Preparation

Download the `mit` and `lsrw` folders (about 900M total) from [here](https://github.com/xiaomengwupx/cone-data) and store them in the `data` folder. These two benchmark datasets have been reorganized by [Ma et al](https://github.com/vis-opt-group/SCI). In this reorganized configuration, both datasets contain 500 training images. The number of test images is 100 and 50, respectively.

As an example, for the MIT dataset, the training data is located in the following folder. Note that the ground-truth images in the `gt` folder are not used.

```
./data/mit/train/input
```

The data for testing and evaluation are located in the following folders.

```
./data/mit/test/input
./data/mit/test/gt
```

## Testing

The following command reproduces the results reported in our paper.

```bash
python test.py
```

We provide three models pre-trained on different datasets with different comparametric equations. They are located under the `exp` folder. See the table below for details.

| Model | Dataset | CEM |
| :--- | :---: | :---: |
| `train-20221108-052232` | MIT | Sigmoid Correction |
| `train-20221108-053222` | LSRW | BetaGamma Correction |
| `train-20221110-055039` | LSRW | Preferred Correction |

By default, `test.py` evaluates the performance of the first model. If, for example, you want to test the second model, you need to change the value of the argument `dataset` to `lsrw` and the value of `model` to `train-20221108-053222`. These values are specified in the next two lines.

