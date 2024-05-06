import os
import numpy as np
import skimage.io as io
import json
import tifffile
from skimage.filters import window
from scipy.ndimage import gaussian_filter
import time

import sys
sys.path.append(r"../opticalaberrations/src/")

from utils import fftconvolution


def read_json(path):
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    return data


# Thayer's code
def fft(image, padsize=None):
    if padsize is not None:
        shape = image.shape[1]
        size = shape * (padsize / shape)
        pad = int((size - shape) // 2)
        image = np.pad(image, ((pad, pad), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    fft_image = np.fft.ifftshift(image)
    fft_image = np.fft.fftn(fft_image)
    fft_image = np.fft.fftshift(fft_image)
    return fft_image


def ifft(fft_image):
    image = np.fft.fftshift(fft_image)
    image = np.fft.ifftn(image)
    image = np.abs(np.fft.ifftshift(image))
    return image


def crop_center(patch, patch_size=64):
    z, y, x = patch.shape
    z = z//2
    y = y//2
    x = x//2

    z_start = z - (patch_size // 2) + 1
    y_start = y - (patch_size // 2) + 1
    x_start = x - (patch_size // 2) + 1
    z_end = z_start+patch_size
    y_end = y_start + patch_size
    x_end = x_start + patch_size

    return patch[z_start:z_end, y_start:y_end, x_start:x_end]


# Thayer's code
def photons2electrons(image, quantum_efficiency=.82):
    return image * quantum_efficiency


def electrons2photons(image, quantum_efficiency=.82):
    return image / quantum_efficiency


def electrons2counts(image, electrons_per_count=.22):
    return image / electrons_per_count


def counts2electrons(image, electrons_per_count=.22):
    return image * electrons_per_count


def photons2counts(image, quantum_efficiency=.82, electrons_per_count=.22):
    return electrons2counts(photons2electrons(image))


def counts2photons(image, quantum_efficiency=.82, electrons_per_count=.22):
    return electrons2photons(counts2electrons(image))


def randuniform(var):
    var = (var, var) if np.isscalar(var) else var

    return np.random.uniform(*var)


def normal_noise(mean: float, sigma: float, size: tuple) -> np.array:
    mean = randuniform(mean)
    sigma = randuniform(sigma)
    return np.random.normal(loc=mean, scale=sigma, size=size).astype(np.float32)


def poisson_noise(image: np.ndarray) -> np.array:
    image = np.nan_to_num(image, nan=0)
    return np.random.poisson(lam=image).astype(np.float32) - image


def noise(image, mean_background_offset=100, sigma_background_noise=40, quantum_efficiency=.82, electrons_per_count=.22):
    """
    Args:
        image: noise-free image in photons
        mean_background_offset: camera background offset
        sigma_background_noise: read noise from the camera
        quantum_efficiency: quantum efficiency of the camera
        electrons_per_count: conversion factor to go from electrons to counts
    Returns:
        noisy image in counts
    """
    image = photons2electrons(image, quantum_efficiency=quantum_efficiency)
    sigma_background_noise *= electrons_per_count  # electrons;  40 counts = 40 * .22 electrons per count
    dark_read_noise = normal_noise(mean=0, sigma=sigma_background_noise, size=image.shape)  # dark image in electrons
    shot_noise = poisson_noise(image)   # shot noise in electrons

    image += shot_noise + dark_read_noise
    image = electrons2counts(image, electrons_per_count=electrons_per_count)

    image += mean_background_offset    # add camera offset (camera offset in counts)
    image[image < 0] = 0
    return image.astype(np.float32)


if __name__ == "__main__":
    root = r"/clusterfs_m/nvme/sayan/AI/testing_128/psf_comp_test_125/mode_14/"

    exp_num = 1
    model_num = 4
    set_num = "testing_set_1"
    dataset_name = "jrc_choroid-plexus-2"
    res_grp = "s2"
    psf_num = "psf6"
    roi_num = "roi10"

    offset = 100

    # path = os.path.join(root, set_num, dataset_name, res_grp, psf_num, roi_num)
    path = root

    ip_path = os.path.join(path, "input/")
    ip_norm_path = os.path.join(path, "input_norm/")
    # gt_norm_path = os.path.join(path, "gt_norm/")
    # ip_norm_path = os.path.join(path, "input/")

    # ip_path = os.path.join(path, "custom_predictions_mito_denoise_drunet_1/", "predictions/")

    gt_path = os.path.join(path, "gt/")

    pred_str = "custom_predictions_mito_drunet128_"
    # pred_str = "iterative_mito_drunet_4_corrected"
    # pred_str = "custom_predictions_mito_drunet_4_stitched"
    pred_path = os.path.join(path, pred_str + str(model_num), "predictions")

    tif_files = []
    json_files = []
    for file in os.listdir(ip_path):
        if file.endswith(".tif"):
            tif_files.append(file)
        else:
            json_files.append(file)

    tif_files.sort()
    json_files.sort()

    ideal_psf_path = r"/clusterfs_m/nvme/sayan/AI/psfs_updated/psf-ideal/YuMB_lambda510/z200-y97-x97/z64-y64-x64/z15/single/photons_5000-10000/amp_0-p0/1.tif"

    psf_path = os.path.join(path, pred_str + str(model_num), "psf-" + str(offset))
    os.makedirs(psf_path, exist_ok=True)

    # pred_psf_path = os.path.join(psf_path, "gt_psf/")
    pred_psf_path = os.path.join(psf_path, "pred_psf/")
    os.makedirs(pred_psf_path, exist_ok=True)

    for i in range(0, len(tif_files)):
        start_time = time.time()
        ip = io.imread(os.path.join(ip_path, tif_files[i]))
        gt = io.imread(os.path.join(gt_path, tif_files[i]))
        pred = io.imread(os.path.join(pred_path, tif_files[i]))


        ideal_psf = io.imread(ideal_psf_path)

        ipsf = ideal_psf.copy()

        pad_size = 128
        start_idx = pad_size // 2 - ipsf.shape[0] // 2
        end_idx = start_idx + ipsf.shape[0]
        z, y, x = pad_size, pad_size, pad_size

        ipsf_pad = np.zeros((pad_size, pad_size, pad_size))
        ipsf_pad[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx] = ipsf
        ipsf = ipsf_pad

        ideal_otf = fft(ipsf/np.sum(ipsf))
        ideal_otf = ideal_otf / np.nanmax(ideal_otf)

        na_mask = np.abs(ideal_otf)
        threshold = np.nanpercentile(na_mask.flatten(), 65)
        na_mask = np.where(na_mask < threshold, na_mask, 1.)
        na_mask = np.where(na_mask >= threshold, na_mask, 0.).astype(bool)


        ip -= offset
        ip[ip < 0] = 0

        ipsf = ipsf / np.sum(ipsf)
        idfft = fft(ipsf)

        preda = pred.copy()

        pfft = fft(pred)
        ## pfft = pfft / np.nanmax(pfft[z // 2 - 2:z // 2 + 2, y // 2 - 2:y // 2 + 2, x // 2 - 2:x // 2 + 2])
        pred = np.abs(ifft(pfft * idfft))
        pred[pred < 0] = 0

        ip = ip / np.sum(ip)
        # gt = gt / np.sum(gt)
        pred = pred / np.sum(pred)

        # w = window(('tukey', .05), ip.shape[1:])
        # w = w[np.newaxis, ...]
        w = window(('tukey', 1), ip.shape)

        # io.imsave("/clusterfs_m/nvme/sayan/AI/testing_million/psf_comp_test/iterative_mito_drunet_4_corrected1/psf-100/tuk_0.tif", w.astype(np.float32))

        ip *= w
        gt *= w
        pred *= w

        # print(tif_files[i], np.count_nonzero(gt)/np.size(gt))
        # io.imsave(os.path.join(psf_path, "ip_tuk_64_" + tif_files[i]), ip.astype(np.float32))
        # io.imsave(os.path.join(psf_path, "gt_tuk_64_" + tif_files[i]), pred.astype(np.float32))
        # io.imsave(os.path.join(psf_path, "gt_tuk_64_" + tif_files[i]), gt.astype(np.float32))

        preda = preda / np.sum(preda)
        preda *= w
        fft_preda = fft(preda)
        fft_preda = fft_preda * na_mask
        fft_preda = fft_preda / np.nanmax(fft_preda[z // 2 - 2:z // 2 + 2, y // 2 - 2:y // 2 + 2, x // 2 - 2:x // 2 + 2])

        fft_ip = fft(ip)
        fft_ip = fft_ip * na_mask
        fft_ip = fft_ip / np.nanmax(fft_ip[z // 2 - 2:z // 2 + 2, y // 2 - 2:y // 2 + 2, x // 2 - 2:x // 2 + 2])


        fft_pred = fft(pred)
        fft_pred = fft_pred * na_mask
        fft_pred = fft_pred / np.nanmax(fft_pred[z // 2 - 2:z // 2 + 2, y // 2 - 2:y // 2 + 2, x // 2 - 2:x // 2 + 2])


        pred_otf = (fft_ip / fft_pred) * ideal_otf
        pred_otf = pred_otf * na_mask
        pred_otf = pred_otf / np.nanmax(pred_otf[z // 2 - 2:z // 2 + 2, y // 2 - 2:y // 2 + 2, x // 2 - 2:x // 2 + 2])
        pred_otf = np.nan_to_num(pred_otf)

        p = pred_otf.copy()
        pred_otf *= (fft_pred * np.conj(fft_pred)) / (fft_pred * np.conj(fft_pred) + 1e-9)
        # pred_otf *= (fft_preda * np.conj(fft_preda)) / (fft_preda * np.conj(fft_preda) + 1e-9)
        pred_otf = pred_otf / np.nanmax(pred_otf[z // 2 - 2:z // 2 + 2, y // 2 - 2:y // 2 + 2, x // 2 - 2:x // 2 + 2])
        pred_otf = np.nan_to_num(pred_otf)


        pred_psf = ifft(pred_otf)

        pred_psf = pred_psf / np.nanmax(pred_psf)
        pred_psf = pred_psf[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]

        # io.imsave(os.path.join(pred_psf_path, "gt_psf_" + tif_files[i]), pred_psf.astype(np.float32))
        io.imsave(os.path.join(pred_psf_path, "pred_psf_" + tif_files[i]), pred_psf.astype(np.float32))

        print(f"Total time elapsed: {time.time() - start_time:.2f} sec.")

        # io.imsave(os.path.join(psf_path, "na_mask.tif"), na_mask.astype(np.float32))
