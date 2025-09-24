import os
import tqdm

import numpy as np
import torch
from PIL import Image
from pywt import dwt2, idwt2
import blobfile as bf
from torch.utils.data import DataLoader, Dataset


def load_data_wavelet(
    data_dir, batch_size, j, conditional=True, deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param j: the scale at which to sample wavelet coefficients (from 1 to J)
    :param conditional: whether to learn the distribution of the low frequencies,
    or the distribution of the high frequencies given the low frequencies
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # Let's put that here because we don't need it when generating the wavelet dataset.
    from mpi4py import MPI

    all_files = _list_image_files_recursively(data_dir)
    dataset = WaveletDataset(
        all_files,
        j=j,
        conditional=conditional,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npz"] and "stats" not in entry:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def wavelet_stats(j, dir_name):
    """ Returns mean and std of each channel of wavelet coefficients, as (12,)-shaped CPU float tensors.
    dir_name is the directory of the wavelet coeffs, which is used to determine which statistics to use. """
    if "cifar" in dir_name:
        if "haar" in dir_name:
            if j == 1:
                mean = torch.tensor([1.5084e-01, 3.5875e-01, 6.0854e-01, 6.9881e-02, 7.1949e-02,
                                     6.8880e-02, -1.4889e-03, -1.3639e-03, -9.8872e-04, 2.5056e+02,
                                     2.4588e+02, 2.2745e+02])
                std = torch.tensor([23.8289, 23.3721, 23.1966, 21.8616, 21.5753, 21.2846, 8.7760,
                                    8.7048, 8.5958, 121.2786, 119.4935, 129.1759])
            elif j == 2:
                mean = torch.tensor([5.9072e-01, 1.4340e+00, 2.4509e+00, 1.1335e-01, 1.4552e-01,
                                     1.2242e-01, -5.0671e-03, -2.8707e-03, -5.6267e-04, 5.0111e+02,
                                     4.9175e+02, 4.5489e+02])
                std = torch.tensor([62.5310, 60.9610, 60.7940, 54.4673, 53.5758, 53.0284, 27.6028,
                                    27.1800, 26.7973, 226.2628, 223.1309, 243.9541])
            elif j == 3:
                mean = torch.tensor([1.9777e+00, 5.4197e+00, 9.6603e+00, 4.7795e-01, 4.3230e-01,
                                     4.7100e-01, -1.3948e-02, -2.5899e-02, -1.0191e-02, 1.0022e+03,
                                     9.8350e+02, 9.0979e+02])
                std = torch.tensor([148.8808, 145.5707, 146.7062, 122.1698, 119.9842, 119.6010, 68.9693,
                                    67.4152, 66.8672, 403.6430, 398.7332, 444.5927])
            elif j == 4:
                mean = torch.tensor([1.3868e+01, 2.6126e+01, 4.2736e+01, 6.6111e-01, 9.9249e-01,
                                     7.9822e-01, -2.5567e-01, -5.0524e-01, -5.4482e-01, 2.0044e+03,
                                     1.9670e+03, 1.8196e+03])
                std = torch.tensor([334.5825, 330.7644, 338.9643, 265.5326, 260.9721, 261.3336, 165.5999,
                                    161.9690, 162.2464, 664.5611, 656.9044, 761.1224])
            else:
                assert False
        elif "db2" in dir_name:
            if j == 1:
                mean = torch.tensor([-1.5084e-01, -3.5875e-01, -6.0854e-01, -6.9881e-02, -7.1949e-02,
                                     -6.8880e-02, -1.4889e-03, -1.3639e-03, -9.8872e-04, 2.5056e+02,
                                     2.4588e+02, 2.2745e+02])
                std = torch.tensor([23.8634, 23.4717, 23.9319, 20.5419, 20.2565, 20.0156, 7.5097,
                                    7.4586, 7.3849, 121.5871, 119.7888, 129.3195])
            elif j == 2:
                mean = torch.tensor([-4.8435e-01, -9.2286e-01, -1.4610e+00, -3.0987e-01, -3.6886e-01,
                                     -3.8047e-01, -1.0900e-02, -8.7935e-03, -4.7146e-03, 5.0111e+02,
                                     4.9175e+02, 4.5489e+02])
                std = torch.tensor([65.9840, 64.5194, 65.7495, 53.8343, 52.8749, 52.2891, 26.8897,
                                    26.4856, 26.1290, 226.1820, 223.0167, 243.2118])
            elif j == 3:
                mean = torch.tensor([-2.4802e+00, -2.5356e+00, -2.8297e+00, -1.5695e+00, -2.1028e+00,
                                     -2.1140e+00, -1.3543e-01, -1.2247e-01, -1.1103e-01, 1.0022e+03,
                                     9.8350e+02, 9.0979e+02])
                std = torch.tensor([160.1916, 156.8015, 161.3618, 121.6037, 119.5449, 118.7682, 70.6232,
                                    69.0162, 68.4789, 398.9839, 394.0735, 437.9070])
            elif j == 4:
                mean = torch.tensor([-17.5323, -12.7797, -8.8121, -3.7333, -8.2556, -9.4273,
                                     -2.5554, -3.1810, -2.9647, 2004.4410, 1967.0073, 1819.5775])
                std = torch.tensor([371.3722, 375.7181, 394.5940, 256.4750, 251.3915, 250.5833, 181.3394,
                                    177.8100, 176.9127, 632.3323, 620.4482, 719.0829])
            else:
                assert False
        elif "periodic" in dir_name:  # (db4)
            if j == 1:
                mean = torch.tensor([-1.6681e-01, -3.7783e-01, -6.3130e-01, -6.7963e-02, -7.0681e-02,
                                     -6.7094e-02, -1.6775e-03, -1.5074e-03, -1.1346e-03, 2.5064e+02,
                                     2.4597e+02, 2.2780e+02])
                std = torch.tensor([19.0516, 18.8084, 18.8768, 17.4892, 17.2984, 17.0540, 6.4141,
                                    6.3870, 6.3243, 123.1284, 121.3336, 130.8136])
            elif j == 2:
                mean = torch.tensor([6.7784e-01, 1.5286e+00, 2.5582e+00, 1.4630e-01, 1.8345e-01,
                                     1.6133e-01, -1.0500e-02, -7.2771e-03, -5.7023e-03, 5.0127e+02,
                                     4.9193e+02, 4.5561e+02])
                std = torch.tensor([66.2519, 64.8669, 66.3352, 53.8556, 52.8813, 52.2979, 26.8423,
                                    26.4495, 26.0996, 229.4159, 226.2329, 246.2227])
            elif j == 3:
                mean = torch.tensor([1.4401e+00, 6.8331e-02, -1.4055e+00, 1.2227e+00, 1.7222e+00,
                                     1.7234e+00, -1.1565e-01, -9.5441e-02, -8.0341e-02, 1.0025e+03,
                                     9.8387e+02, 9.1121e+02])
                std = torch.tensor([143.4349, 139.2118, 140.1205, 116.3783, 114.4589, 113.8165, 68.1628,
                                    66.6236, 66.0399, 414.4386, 409.6375, 453.3744])
            elif j == 4:
                mean = torch.tensor([-9.9738e+00, -2.0333e+01, -3.4489e+01, 7.0531e-01, 1.2134e+00,
                                     1.5466e+00, -2.5195e-01, -5.8145e-01, -4.9888e-01, 2.0051e+03,
                                     1.9677e+03, 1.8224e+03])
                std = torch.tensor([351.1261, 344.7343, 350.3575, 289.1377, 284.6882, 284.9703, 188.9330,
                                    184.4474, 185.0562, 666.5969, 660.9738, 763.4114])
            else:
                assert False
        else:  # Statistics of wavelet coefficients with symmetric border conditions.
            if j == 1:
                mean = torch.tensor([-3.3138e-03, -5.5262e-03, -5.5076e-03, 2.7390e-02, 2.5372e-02,
                                     2.4046e-02, -3.3241e-04, -3.3805e-04, -1.4877e-04, 2.5237e+02,
                                     2.5014e+02, 2.3304e+02])
                std = torch.tensor([15.8468, 15.6330, 15.4486, 14.8652, 14.7261, 14.4775, 5.5330,
                                    5.5135, 5.4578, 126.3216, 124.8193, 135.5778])
            elif j == 2:
                mean = torch.tensor([1.6527e-01, 1.8018e-01, 2.2636e-01, 2.6852e-01, 3.1685e-01,
                                     3.2989e-01, -9.8129e-04, 2.3386e-04, 2.0201e-05, 5.0838e+02,
                                     5.0980e+02, 4.7787e+02])
                std = torch.tensor([51.2353, 49.8825, 49.4493, 44.4847, 43.8424, 43.5495, 22.9943,
                                    22.6914, 22.3414, 249.1176, 246.5352, 271.4594])
            elif j == 3:
                mean = torch.tensor([-8.3363e-01, -1.3979e+00, -2.0309e+00, -7.4508e-02, -5.4218e-01,
                                     -6.7501e-01, -1.7483e-02, -2.5347e-02, -3.0483e-02, 1.0226e+03,
                                     1.0353e+03, 9.7609e+02])
                std = torch.tensor([108.2199, 105.0126, 105.7366, 87.2165, 85.9143, 86.1309, 47.6882,
                                    46.6108, 46.1700, 493.4673, 488.6843, 543.3653])
            elif j == 4:
                mean = torch.tensor([1.7126e+00, 3.3977e+00, 4.6433e+00, -1.0988e+00, -1.9746e-01,
                                     1.9132e-02, -7.5810e-03, -6.3298e-02, -5.3144e-02, 2.0566e+03,
                                     2.0982e+03, 1.9921e+03])
                std = torch.tensor([204.3679, 200.9428, 203.5329, 160.3592, 157.2399, 159.0522,
                                    69.1469, 67.6269, 67.2288, 1004.1478, 995.1146, 1111.2657])
            else:
                assert False
    elif "celebA128" in dir_name:
        if "haar" in dir_name:
            if j == 1:
                mean = torch.tensor([-2.0805e-02, 5.6151e-02, 9.0666e-02, 1.1676e-02, 1.8160e-02,
                                     2.1746e-02, 3.4420e-04, 2.5812e-04, 2.7992e-04, 2.6360e+02,
                                     2.1249e+02, 1.8524e+02])
                std = torch.tensor([12.5765, 12.3731, 12.1788, 14.5429, 14.2343, 14.0763, 5.3062,
                                    5.3050, 5.2848, 151.4951, 136.9868, 134.5582])
            elif j == 2:
                mean = torch.tensor([4.2190e-03, 3.2070e-01, 4.6083e-01, 6.5188e-02, 9.0771e-02, 1.0497e-01,
                                     1.7535e-03, 1.9244e-03, 1.5806e-03, 5.2720e+02, 4.2498e+02, 3.7047e+02])
                std = torch.tensor([35.3143, 34.1401, 33.2296, 41.1878, 39.4662, 38.8329, 15.5241,
                                    15.4770, 15.3383, 297.6886, 268.5120, 263.7726])
            elif j == 3:
                mean = torch.tensor([4.5776e-01, 1.8720e+00, 2.4723e+00, 2.0857e-02, 1.2023e-01, 1.8223e-01,
                                     7.2644e-02, 7.0623e-02, 6.5009e-02, 1.0544e+03, 8.4996e+02, 7.4095e+02])
                std = torch.tensor([94.3447, 88.0583, 84.2734, 116.4416, 107.6941, 105.1277, 44.2667,
                                    42.8449, 42.1394, 574.5038, 516.9172, 508.2992])
            elif j == 4:
                mean = torch.tensor([3.0280e+00, 7.8968e+00, 1.0708e+01, -1.0779e-01, 5.6080e-01,
                                     8.5929e-01, 2.9388e-01, 2.2086e-01, 2.0036e-01, 2.1088e+03,
                                     1.6999e+03, 1.4819e+03])
                std = torch.tensor([242.9951, 216.7215, 204.7005, 316.2409, 290.9759, 284.7321,
                                    133.2096, 122.0385, 116.2106, 1069.3032, 960.3240, 947.0361])
            else:
                assert False
        elif "db2" in dir_name:
            if j == 1:
                mean = torch.tensor([2.0827e-02, -5.6266e-02, -9.0733e-02, -1.1423e-02, -1.7809e-02,
                                     -2.1415e-02, 4.0263e-04, 3.2171e-04, 3.4907e-04, 2.6361e+02,
                                     2.1254e+02, 1.8530e+02])
                std = torch.tensor([13.3895, 12.8722, 12.8571, 13.0829, 12.9284, 13.0542, 4.8509,
                                    4.8344, 4.8384, 151.5620, 137.0760, 134.6091])
            elif j == 2:
                mean = torch.tensor([7.5054e-02, -9.2360e-02, -1.7778e-01, -1.4502e-01, -1.6591e-01,
                                     -1.9130e-01, -1.3615e-02, -1.3226e-02, -1.3668e-02, 5.2722e+02,
                                     4.2508e+02, 3.7061e+02])
                std = torch.tensor([36.9497, 35.5440, 35.0142, 36.8797, 35.9000, 35.7512, 14.3960,
                                    14.2802, 14.2355, 298.2474, 269.0784, 264.1430])
            elif j == 3:
                mean = torch.tensor([-1.7188e+00, -1.6256e+00, -1.5756e+00, 1.9950e-01, 2.3563e-01,
                                     2.4929e-01, 8.1207e-02, 8.3861e-02, 8.8625e-02, 1.0544e+03,
                                     8.5017e+02, 7.4121e+02])
                std = torch.tensor([97.7676, 92.1623, 89.1650, 104.8405, 99.8985, 98.2133, 42.2997,
                                    41.3491, 40.6341, 577.4632, 519.0633, 509.7414])
            elif j == 4:
                mean = torch.tensor([-1.9724e+00, -5.2756e-01, -1.9344e+00, 4.6028e-01, -3.5897e+00,
                                     -4.7755e+00, 8.8264e-01, 1.5908e+00, 1.6094e+00, 2.1089e+03,
                                     1.7003e+03, 1.4824e+03])
                std = torch.tensor([239.5857, 223.0819, 216.1101, 294.6428, 273.9173, 268.0204,
                                    122.6664, 115.3752, 111.3737, 1083.7838, 969.3211, 953.0862])
            else:
                assert False
        else:  # db4
            if j == 1:
                mean = torch.tensor([2.0680e-02, -5.6395e-02, -9.0897e-02, -1.1634e-02, -1.8046e-02,
                                     -2.1721e-02, 3.3215e-04, 2.4775e-04, 2.5438e-04, 2.6361e+02,
                                     2.1252e+02, 1.8527e+02])
                std = torch.tensor([9.9334, 9.7313, 9.7180, 10.7857, 10.7543, 10.7778, 4.1779,
                                    4.1745, 4.1710, 152.0460, 137.5670, 135.1047])
            elif j == 2:
                mean = torch.tensor([-4.8471e-03, 3.1113e-01, 4.5230e-01, 7.9938e-02, 1.0621e-01,
                                     1.2299e-01, 4.4224e-03, 4.6449e-03, 4.2846e-03, 5.2723e+02,
                                     4.2504e+02, 3.7054e+02])
                std = torch.tensor([36.6215, 35.3167, 35.0088, 35.3226, 34.6283, 34.7474, 14.1558,
                                    14.0424, 14.0414, 299.4707, 270.2868, 265.2974])
            elif j == 3:
                mean = torch.tensor([1.4655e+00, 8.0662e-01, 5.1352e-01, -2.0735e-01, -2.8336e-01,
                                     -3.2595e-01, -1.1257e-01, -1.1134e-01, -1.0586e-01, 1.0545e+03,
                                     8.5008e+02, 7.4107e+02])
                std = torch.tensor([77.0246, 74.3153, 71.8557, 93.2029, 88.0841, 86.2351, 40.9821,
                                    40.4778, 39.9922, 585.1747, 526.5920, 517.0410])
            elif j == 4:
                mean = torch.tensor([-3.3001e+00, -7.3695e+00, -9.4369e+00, 7.0316e-02, 2.5514e-01,
                                     2.2990e-01, 3.5150e-01, 5.0090e-01, 5.4376e-01, 2.1089e+03,
                                     1.7002e+03, 1.4821e+03])
                std = torch.tensor([238.3555, 216.2203, 203.2365, 278.3469, 258.7746, 251.4253,
                                    133.6246, 123.4415, 116.6996, 1103.4314, 990.0447, 975.2424])
            else:
                assert False
    elif "celebA256" in dir_name:  # periodic, haar
        if j == 1:
            mean = torch.tensor([-1.8950e-02, 2.6135e-02, 4.5120e-02, 7.8701e-03, 9.9624e-03,
                                 1.1365e-02, -2.5411e-05, 1.3390e-06, -1.2305e-05, 2.6358e+02,
                                 2.1243e+02, 1.8517e+02])
            std = torch.tensor([9.4817, 9.3714, 9.2297, 11.0562, 10.8966, 10.7943, 4.5293,
                                4.5283, 4.5148, 153.1306, 138.4616, 135.9259])
        elif j == 2:
            mean = torch.tensor([-6.6196e-02, 1.1833e-01, 1.9571e-01, 2.6782e-02, 3.5487e-02,
                                 4.1759e-02, -1.6703e-03, -1.9025e-03, -1.6935e-03, 5.2715e+02,
                                 4.2485e+02, 3.7035e+02])
            std = torch.tensor([25.8843, 25.2148, 24.5008, 29.6553, 28.6981, 28.2404, 11.9349,
                                11.9228, 11.8371, 303.4866, 274.0163, 269.0083])
        elif j == 3:
            mean = torch.tensor([-1.7773e-01, 6.9659e-01, 1.0385e+00, 1.3991e-01, 1.7954e-01,
                                 2.0260e-01, 1.0458e-02, 8.6989e-03, 1.0048e-02, 1.0543e+03,
                                 8.4971e+02, 7.4069e+02])
            std = torch.tensor([72.4755, 68.0678, 64.7857, 83.8403, 78.6150, 76.6667, 31.7745,
                                31.0503, 30.5978, 595.9235, 537.1792, 527.6832])
        elif j == 4:
            mean = torch.tensor([9.7705e-01, 3.7093e+00, 5.1309e+00, 4.4421e-02, 2.1589e-01, 3.4698e-01,
                                 1.3532e-01, 1.4612e-01, 1.4798e-01, 2.1086e+03, 1.6994e+03, 1.4814e+03])
            std = torch.tensor([189.5771, 176.1225, 167.5214, 233.5511, 215.3104, 209.6427,
                                90.4419, 85.7787, 82.6287, 1149.7099, 1034.1647, 1017.3156])
        else:
            assert False
    elif "celebA512" in dir_name:  # periodic, haar
        if j == 1:
            mean = torch.tensor([-1.1363e-02, 1.3890e-02, 2.4776e-02, 4.3457e-03, 5.3198e-03,
                                 5.7553e-03, -2.2983e-06, -1.2800e-06, -8.3599e-07, 2.6366e+02,
                                 2.1248e+02, 1.8526e+02])
            std = torch.tensor([7.2286, 7.1915, 7.1149, 8.5327, 8.4692, 8.4049, 3.3021,
                                3.3003, 3.2946, 153.9831, 139.3323, 136.7681])
        elif j == 2:
            mean = torch.tensor([-4.3891e-02, 5.8089e-02, 1.0299e-01, 1.6098e-02, 2.0212e-02,
                                 2.2092e-02, -5.9989e-05, -3.8128e-05, -1.3797e-05, 5.2732e+02,
                                 4.2496e+02, 3.7052e+02])
            std = torch.tensor([19.4262, 19.1882, 18.7960, 22.4126, 22.0127, 21.7107, 10.0399,
                                10.0492, 9.9899, 306.3703, 276.9480, 271.8411])
        elif j == 3:
            mean = torch.tensor([-1.7971e-01, 2.4836e-01, 4.3594e-01, 5.3414e-02, 7.0277e-02,
                                 7.7890e-02, -2.9127e-03, -3.1856e-03, -2.7044e-03, 1.0546e+03,
                                 8.4993e+02, 7.4103e+02])
            std = torch.tensor([52.1295, 50.4405, 48.5102, 59.8510, 57.2787, 55.9308, 24.1832,
                                24.0358, 23.7849, 607.0967, 548.0857, 538.0918])
        elif j == 4:
            mean = torch.tensor([-5.6173e-01, 1.4573e+00, 2.3674e+00, 2.5329e-01, 3.3222e-01,
                                 3.7620e-01, 1.3555e-02, 2.1216e-02, 2.5543e-02, 2.1093e+03,
                                 1.6999e+03, 1.4821e+03])
            std = torch.tensor([145.4836, 136.1626, 129.0343, 168.1741, 157.3670, 153.0042,
                                64.4688, 62.3135, 60.3908, 1191.9149, 1074.4310, 1055.6791])
        else:
            assert False
    else:
        assert False
    return mean, std


class WaveletDataset(Dataset):
    def __init__(self, wav_paths, j, conditional, shard=0, num_shards=1):
        super().__init__()
        self.local_wav = wav_paths[shard:][::num_shards]
        self.j = j
        self.conditional = conditional
        self.mean, self.std = wavelet_stats(self.j, wav_paths[0])

    def __len__(self):
        return len(self.local_wav)

    def __getitem__(self, idx):
        path = self.local_wav[idx]
        npz_dict = np.load(path)  # "j{j}" -> (12, Nj, Nj) numpy array of wavelet coefficients (j ranges from 1 to J).
        out = torch.from_numpy(npz_dict[f"j{self.j}"]).float()

        out -= self.mean[:, None, None]
        out /= self.std[:, None, None]

        out_high = out[0:9]
        out_low = out[9:12]

        out_dict = {}
        if self.conditional:
            target = out_high
            out_dict["conditional"] = out_low
        else:
            target = out_low

        return target, out_dict


""" This module, when ran as a script, also serves to generate a dataset of wavelet coefficients. """


def generate_wav_dataset(image_dir, wav_dir, J, wavelet, border_condition):
    """ Computes the wavelet coefficients of the images in image_dir and dumps them in wav_dir. """
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    bar = tqdm.tqdm(os.listdir(image_dir))
    for file in bar:
        if not(os.path.splitext(file)[-1].lower() in ['.jpg', '.png']):
            continue
        img_filename = os.path.join(image_dir, file)
        wav_coeffs = image_to_wavelets(img_filename, J, wavelet, border_condition)
        bar.desc = f"Shapes {[wav_coeffs[j].shape for j in range(J)]}"
        wavelet_filename = os.path.join(wav_dir, os.path.splitext(file)[0] + ".npz")
        np.savez(wavelet_filename, **{f"j{j+1}": coeffs for j, coeffs in enumerate(wav_coeffs)})


def image_to_wavelets(img_fname, J, wavelet, border_condition):
    """
    Outputs (12, W', H') where W,H are not necessarily W/2 or H/2; it depends on the chosen signal extension mode
    (default = symmetric).
    With mode=periodization, W'=W/2 and H'=H/2 but this mode might not be well-suited for natural images.

    Wavelet coefficients are stacked for each channel, ie the 3 first channels are the details of the R channel,
    the 3 next are for the G channel, etc. The last 3 channels are the low-frequencies.
    """
    with Image.open(img_fname) as im:
        img_array = np.moveaxis(np.array(im), -1, 0)

    rets = []  # j -> coeffs (Nj, Nj, 12)
    for j in range(J):
        phi, (psiH, psiV, psiD) = dwt2(img_array, wavelet=wavelet, mode=border_condition)
        ret = np.concatenate((psiH, psiV, psiD, phi), axis=-3)
        rets.append(ret)
        img_array = phi
    return rets


def wavelet_to_image(img_array, border_condition, wavelet, output_size=None):
    """ (*, 12, H', W') -> (*, 3, H, W) float arrays.
    Output size is used because the reconstructed image is sometimes too big. """
    phi = img_array[..., 9:12, :, :]
    psiH = img_array[..., :3, :, :]
    psiV = img_array[..., 3:6, :, :]
    psiD = img_array[..., 6:9, :, :]
    ret = idwt2((phi, (psiH, psiV, psiD)), wavelet=wavelet, mode=border_condition)
    if output_size is not None:
        ret = ret[..., :output_size, :output_size]
    return ret  # (*, 3, H, W)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", help="directory of original images")
    parser.add_argument("--wav-dir", help="directory to output wavelet coefficients")
    parser.add_argument("--J", type=int, default=4, help="maximum scale of the wavelet decomposition")
    parser.add_argument("--wavelet", default="db4", help="wavelet to use")
    parser.add_argument("--border-condition", default="symmetric", help="signal extension mode for wavelet transform")
    args = parser.parse_args()

    generate_wav_dataset(args.image_dir, args.wav_dir, args.J, args.wavelet, args.border_condition)

