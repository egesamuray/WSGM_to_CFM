# Wavelet and Curvelet Score-Based Generative Models

This repository contains the codebase for training and sampling from both wavelet-based and curvelet-based Score-Based Generative Models. It extends the original diffusion model implementation from OpenAI’s **improved-diffusion** repository:contentReference[oaicite:6]{index=6} (see LICENSE) to use multi-scale wavelet or curvelet representations of images.

## Installation

Clone this repository and navigate into it, then run:

```bash
pip install -e .
This will install the improved_diffusion Python package that the scripts depend on.

Preparing Data

For wavelet models: download the CelebA-HQ dataset (e.g., from this source), then generate wavelet coefficients using:

python improved_diffusion/wavelet_datasets.py --image-dir /path/to/celeba_hq \
    --wav-dir /path/to/wavelet/output --J NUMBER_OF_SCALES \
    --border-condition periodization --wavelet haar


For curvelet models: no precomputation is required. The dataset images are used directly, and curvelet coefficients are computed on-the-fly by the data loader.

Training and Sampling

You can use the run_exps.py script to train and sample models. For example, to train curvelet-based models, run:

python run_exps.py --task curvelet --data_dir /path/to/celeba_hq --j 1 --final_size 64


This will train two models for scale j=1 (finest scale) at 64×64 resolution: an unconditional coarse model and a conditional high-frequency (wedge) model. Training logs and checkpoints are saved to a logs/ directory by default. (The script uses MPI with 4 GPUs by default – you may adapt the commands or run on a single GPU by omitting MPI.)

After training, you can generate samples using the unified sampling script:

python scripts/curvelet_generate.py --data_dir /path/to/celeba_hq \
    --coarse_model logs/.../coarse_model.pt --cond_model logs/.../cond_model.pt \
    --j 1 --image_size 64 --num_samples 30000 --output_dir samples_j1


This will first sample low-frequency coarse images from the coarse model, then sample high-frequency details from the conditional model, and finally combine them using the inverse curvelet transform to produce 30,000 full 64×64 images (saved in samples_j1). Adjust --j, --image_size, and the model paths as needed for your experiment. You can also specify --angles_per_scale if you used a non-default wedge configuration during training.

For wavelet models, the process is similar (use --task wavelet in run_exps.py). After training, use the corresponding wavelet sampling script or run_exps.py to generate samples.

FID Computation

Once all models have been trained and you have generated a set of samples (e.g., 30,000 samples for each setting), you can compute the Fréchet Inception Distance (FID) scores. For wavelet models, use:

python get_fid.py


(This script will look for generated samples in the default output locations. If different, provide the correct paths inside the script or via arguments.)

For curvelet models, you can similarly compute FIDs by pointing the FID script to the directory of generated curvelet samples.
