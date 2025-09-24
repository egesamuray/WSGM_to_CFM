from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
import torchvision.transforms as TF
from PIL import Image
from argparse import ArgumentParser
import json

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

__version__ = "retrain"  # Change to recompute stats.


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class WaveletDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths, j):
        super().__init__()
        self.local_wav = wav_paths
        self.j = j

    def __len__(self):
        return len(self.local_wav)

    def __getitem__(self, idx):
        path = self.local_wav[idx]
        npz_dict = np.load(path)  # "j{j}" -> (12, Nj, Nj) numpy array of wavelet coefficients (j ranges from 1 to J).
        out = torch.from_numpy(npz_dict[f"j{self.j}"]).float()
        return out[9:12]


def tensor_summary_stats(x):
    """ Returns summary statistics about x: shape, mean, std, min, max. """
    return f"shape {x.shape}, values in [{x.min():.3f}, {x.max():.3f}] and around {x.mean():.3f} +- {x.std():.3f}"


def get_activations(num_images, dataloader, model, dims=2048, device='cpu'):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    pred_arr = np.empty((num_images, dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        if isinstance(batch, list):
            batch = batch[0]  # TensorDataset...
        batch = batch.to(device)

        # Map to [0, 255], quantize, then map to [0, 1]
        batch -= batch.min()
        batch *= 255 / batch.max()
        batch = batch.int().float() / 255
        if start_idx == 0:
            print(f"Got batch with stats {tensor_summary_stats(batch)}")

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(num_images, dataloader, model, dims=2048, device='cpu'):
    """Calculation of the statistics used by the FID.
    Params:
    -- images       : (N, C, H, W) float tensor with values in [0, 1]
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(num_images, dataloader, model, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1, j=None):
    """ Returns numpy arrays for mu and sigma, taking care of caching.
    path can either be a folder of images, or a numpy array of samples.
    """
    path = Path(path)
    name = f"stats_{__version__}"
    if j is not None:
        name = f"{name}j{j}"
    if path.is_dir():
        cache_file = path / f"{name}.npz"
    else:
        cache_file = path.parent / f"{name}_{path.name}"

    if cache_file.exists():
        f = np.load(str(cache_file))
        return f["mu"], f["sigma"]
    else:
        if j is not None:
            files = sorted([file for ext in ["npy", "npz"]
                            for file in path.glob('*.{}'.format(ext)) if "stats" not in str(file)])
            num_images = len(files)
            dataset = WaveletDataset(files, j=j)
        elif path.is_dir():
            files = sorted([file for ext in IMAGE_EXTENSIONS
                            for file in path.glob('*.{}'.format(ext))])
            num_images = len(files)
            dataset = ImagePathDataset(files, transforms=TF.ToTensor())
        else:
            f = np.load(str(path))
            images = torch.from_numpy(f["arr_0"].transpose(0, 3, 1, 2)).float() / 255  # (N, C, H, W) float [0, 1] tensor
            num_images = images.shape[0]
            dataset = torch.utils.data.TensorDataset(images)

        if batch_size > num_images:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = num_images
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=num_workers)

        mu, sigma = calculate_activation_statistics(num_images, dataloader, model, dims, device)
        np.savez(cache_file, mu=mu, sigma=sigma)
        return mu, sigma


def compute_FID(ref_path, synthetic_paths, device, num_workers=1, batch_size=128, dims=2048, save_path="fid", j=None):
    """Computes FID between reference dataset and synthetic datasets list.
    Paths can either be folders of images, or numpy arrays of samples.
    :param ref_path: path of folder containing reference dataset. Must contain list of images in .jpg or .png format.
    :param synthetic_paths: list of paths of image folders.
    :param device:
    :param num_workers:
    :param batch_size:
    :param dims: dimension used for the Inception network
    :return: a dictionary with keys = the synthetic paths and value = the corresponding FID wrt the reference.
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    mean_ref, cov_ref = compute_statistics_of_path(ref_path, model, batch_size, dims, device, num_workers, j=j)
    fid = {}
    for path in synthetic_paths:
        mean_path, cov_path = compute_statistics_of_path(path, model, batch_size, dims, device, num_workers)
        fid[path] = calculate_frechet_distance(mean_ref, cov_ref, mean_path, cov_path)

    print(fid)
    with open(save_path, 'w') as f:
        json.dump(fid, f)

    return fid


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size to use')
    parser.add_argument('--num-workers', type=int, default=10, help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default="cuda", help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048, choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('paths', type=str, nargs='+', help=('Paths to the generated images or to .npz statistic files. First path is reference path'))
    parser.add_argument('--dump', type=str, help="path of file to dump fids")
    parser.add_argument('--j', default=None, type=int, help="optional scale for wavelet dataset reference")
    args = parser.parse_args()
    compute_FID(args.paths[0], args.paths[1:], args.device, args.num_workers, args.batch_size, args.dims, args.dump, j=args.j)
