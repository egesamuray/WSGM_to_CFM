# Wavelet Score-Based Generative Models

This is the codebase for training and sampling from Wavelet Score-based Generative Models. Most of it is a copy of https://github.com/openai/improved-diffusion (see LICENSE).

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

## Preparing Data

Download the CelebA-HQ dataset (e.g., from here: https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download). You then need to generate wavelet coefficients using:

```
python improved_diffusion/wavelet_datasets.py --image-dir PATH/TO/CELEBA --wav-dir PATH/TO/WAVELET/OUTPUT --J NUMBER_OF_SCALES --border-condition per
iodization --wavelet haar
```

## Training and Sampling

When run, the `run_exps.py` file launches experiments for training and sampling from the models presented in section 4.2 of the paper. You can run `python run_exps.py --print` to see the commands that will be ran, and `python run_exps.py` to actually run them. The commands expect to be piped through MPI with 4 GPUs (e.g., a batch size of 32 on each GPU leads to a total batch size of 128), you should adapt the commands for your system. You should also provide the correct path for the training datasets. Training logs and checkpoints go into a `logs/` folder in the project root directory by default.

## FID computation

Once all models have been trained and 30,000 samples have been generated with several number of discretization steps, run `python get_fid.py` to compute FIDs for every setting. If different, you will need to provide the correct path for the training datasets.

