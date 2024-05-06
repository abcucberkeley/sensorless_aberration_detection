import sys
import logging
sys.path.append(r"../opticalaberrations/src/")
import cli

import os
from utils import fftconvolution
import skimage.io as io
import numpy as np
from skimage.util.shape import view_as_windows
from skimage.transform import resize
from scipy import ndimage
import copy
import time
from itertools import combinations
import json
from scipy.ndimage import rotate, map_coordinates
from typing import Any, Union, Optional


def psf_file_paths(psf_root):
    psf_path_list = []
    stop_str = "amp"
    for root, dirs, files in os.walk(psf_root):
        for name in dirs:
            if stop_str in name:
                psf_path_list.append(os.path.join(root, name))

    psf_path_list.sort(key=lambda x: (x.rsplit('/', 3)[1].lower(), int((x.rsplit('/', 3)[2])[8:].split('-')[0]), float((x.rsplit('/', 3)[3])[4:].split('-')[0].replace('p', '.'))))

    return psf_path_list


# Thayer's code
def rotate_coords(
    shape: Union[tuple, np.ndarray],
    digital_rotations: float,
    axes: tuple = (-2, -1),
):
    dtype = np.float32
    rotations = [digital_rotations]  # np.linspace(0, 360, digital_rotations)

    # Add statement for different axes?
    coords = np.array(
            np.meshgrid(
                np.arange(shape[0]) + .5,
                np.arange(shape[1]) + .5,  # when size is even: FFT is offcenter by 1, and we rotate about pixel seam.
                np.arange(shape[2]) + .5,  # when size is even: FFT is offcenter by 1, and we rotate about pixel seam.
                indexing='ij'
            )
    )
    all_coords = np.zeros((3, 1, *shape), dtype=dtype)

    for i, angle in enumerate(rotations):
        all_coords[:, i] = rotate(
                coords,
                angle=angle,
                reshape=False,
                axes=axes,
                output=dtype,  # output will be floats
                prefilter=False,
                order=1,
        )

    # print(all_coords.shape)

    # # Again, rotation by zero degrees doesn't always become a "no operation". We need to enforce that between emb
    # # dimension otherwise, we will mix between the six embeddings.
    # for emb in range(shape[0]):
    #     all_coords[0, :, emb, :, :] = emb

    return all_coords   # 3 coords (z,y,x), 361 angles, 6 emb, height of emb, width of emb


def rotate_embeddings(emb, angle, axes=(-2, -1), plot: Any = None, debug_rotations: bool = False,
):

    coordinates = rotate_coords(shape=emb.shape, digital_rotations=angle, axes=axes)

    emb = map_coordinates(emb, coordinates=coordinates, output=np.float32, order=1, prefilter=False)

    return emb[0]


def rotate_uniform(patch_list, mode="multi"):
    rotated_patches, rotated_patches_info = [], []

    if mode == "same":
        xy_angle, yz_angle, zx_angle = np.random.choice(180, 3)
        for patch in patch_list:
            rotate_patch_xy = rotate_embeddings(patch, angle=xy_angle, axes=(-2, -1))
            rotate_patch_xy_yz = rotate_embeddings(rotate_patch_xy, angle=yz_angle, axes=(-3, -2))
            rotate_patch_xy_yz_zx = rotate_embeddings(rotate_patch_xy_yz, angle=zx_angle, axes=(-1, -3))
            # rotate_patch_xy = ndimage.rotate(patch, angle=xy_angle, axes=(2, 1), reshape=False)
            # rotate_patch_xy_yz = ndimage.rotate(rotate_patch_xy, angle=yz_angle, axes=(1, 0), reshape=False)
            # rotate_patch_xy_yz_zx = ndimage.rotate(rotate_patch_xy_yz, angle=zx_angle, axes=(0, 2), reshape=False)

            rotated_patches.append(rotate_patch_xy_yz_zx)
            rotated_patches_info.append((float(xy_angle), float(yz_angle), float(zx_angle)))

    elif mode == "multi":
        for patch in patch_list:
            xy_angle, yz_angle, zx_angle = np.random.choice(180, 3)
            rotate_patch_xy = rotate_embeddings(patch, angle=xy_angle, axes=(-2, -1))
            rotate_patch_xy_yz = rotate_embeddings(rotate_patch_xy, angle=yz_angle, axes=(-3, -2))
            rotate_patch_xy_yz_zx = rotate_embeddings(rotate_patch_xy_yz, angle=zx_angle, axes=(-1, -3))
            # rotate_patch_xy = ndimage.rotate(patch, angle=xy_angle, axes=(2, 1), reshape=False)
            # rotate_patch_xy_yz = ndimage.rotate(rotate_patch_xy, angle=yz_angle, axes=(1, 0), reshape=False)
            # rotate_patch_xy_yz_zx = ndimage.rotate(rotate_patch_xy_yz, angle=zx_angle, axes=(0, 2), reshape=False)

            rotated_patches.append(rotate_patch_xy_yz_zx)
            rotated_patches_info.append((float(xy_angle), float(yz_angle), float(zx_angle)))

    return rotated_patches, rotated_patches_info


def axial_voxel_resize(patch_list):
    resized_patches = []
    for patch in patch_list:
        resized_patches.append(resize(patch, (int((patch.shape[0] * .097) / .2), patch.shape[1], patch.shape[2]), order=3, anti_aliasing=True, preserve_range=True))

    return resized_patches


def crop_center(patch_list, patch_size=64):
    cropped_patches = []

    for patch in patch_list:
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

        cropped_patches.append(patch[z_start:z_end, y_start:y_end, x_start:x_end])

    return cropped_patches


# def check_occupancy(patch, occ_thresh=0.02, patch_size=64):
#     z, y, x = patch.shape
#     z = z // 2
#     y = y // 2
#     x = x // 2
#
#     z_start = z - (patch_size // 2) + 1
#     y_start = y - (patch_size // 2) + 1
#     x_start = x - (patch_size // 2) + 1
#     z_end = z_start + patch_size
#     y_end = y_start + patch_size
#     x_end = x_start + patch_size
#     center_patch = patch[z_start:z_end, y_start:y_end, x_start:x_end]
#
#     if (np.count_nonzero(center_patch) / center_patch.size) >= occ_thresh:
#         return True
#     else:
#         return False


def check_occupancy(patch, occ_thresh=0.02, patch_size=64):
    center_patch = patch

    if (np.count_nonzero(center_patch) / center_patch.size) >= occ_thresh:
        return True
    else:
        return False


def read_json(path):
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    return data


def write_json(path, data):
    file = open(path, 'w')
    json.dump(data, file, indent=4)
    file.close()


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


def process_patch(dataset_name, res_grp, file_idx, label_patch_images, label_patch_metadata, label, seed, num_rotations, rotation_mode, sample_patch_size, occupancy_thresh, num_psfs_per_bin, psf_path_list, mean_background_offset, sigma_background_noise, root, ph, num_comp_labels=1):
    # gt_path = os.path.join(root, "training_million/mito_training_set/", dataset_name, res_grp, "gt/")
    # os.makedirs(gt_path, exist_ok=True)
    #
    # input_path = os.path.join(root, "training_million/mito_training_set/", dataset_name, res_grp, "input/")
    # os.makedirs(input_path, exist_ok=True)

    psf_num = 1
    roi_num = 14
    gt_path = os.path.join(root, "testing_million/testing_set_1/", dataset_name, res_grp, "psf"+str(psf_num), "roi"+str(roi_num), "gt/")
    os.makedirs(gt_path, exist_ok=True)

    input_path = os.path.join(root, "testing_million/testing_set_1/", dataset_name, res_grp, "psf"+str(psf_num), "roi"+str(roi_num), "input/")
    os.makedirs(input_path, exist_ok=True)

    info_dict = {}

    label_images = label_patch_images

    i = 0
    info_dict['dataset_name'] = label_patch_metadata[i]['dataset_name']
    info_dict['resolution_group'] = label_patch_metadata[i]['resolution_group']
    info_dict['order'] = label_patch_metadata[i]['order']
    info_dict['voxel_resolution'] = label_patch_metadata[i]['voxel_resolution']

    info_dict['label'] = label

    label_list = info_dict['label'].split('-')

    info_dict['dimensions'] = label_patch_metadata[i]['dimensions']
    info_dict['augmentation_patch_size'] = label_patch_metadata[i]['augmentation_patch_size']
    info_dict['step_size'] = label_patch_metadata[i]['step_size']
    info_dict['sample_patch_size'] = (sample_patch_size, sample_patch_size, sample_patch_size)
    info_dict['photons'] = label_patch_metadata[i]['photons']
    info_dict['seed'] = seed
    info_dict['root_file_index'] = file_idx
    info_dict['patch_info'] = label_patch_metadata[i]['patch_info']
    info_dict['label_intensity_variation_function'] = label_patch_metadata[i]['label_intensity_variation_function']

    for gt_rot_count in range(1, num_rotations+1):
        # rotated_patches, rotation_info = rotate_uniform(label_images, rotation_mode)
        rotated_patches, rotation_info = label_images, [[0.0, 0.0, 0.0]] * len(label_images)

        info_dict['rotation_angles'] = {}
        for j in range(len(rotation_info)):
            info_dict['rotation_angles']['xy_angle_' + label_list[j]] = rotation_info[j][0]
            info_dict['rotation_angles']['yz_angle_' + label_list[j]] = rotation_info[j][1]
            info_dict['rotation_angles']['zx_angle_' + label_list[j]] = rotation_info[j][2]

        resized_patches = axial_voxel_resize(rotated_patches)

        cropped_patches = crop_center(resized_patches, sample_patch_size)

        # final_patch = resized_patches[0]
        # for patch in resized_patches[1:]:
        #     final_patch = np.maximum(final_patch, patch)

        final_patch = cropped_patches[0]
        for patch in cropped_patches[1:]:
            final_patch = np.maximum(final_patch, patch)

        final_patch[final_patch < 1e-5] = 0
        factor = 1
        final_patch *= factor
        ph = label_patch_metadata[i]['photons']
        ph *= factor

        if check_occupancy(final_patch, occupancy_thresh, sample_patch_size):
            # logger.info(f"{info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph)} processing.")

            # cropped_gt_patch = crop_center([final_patch], sample_patch_size)
            # cropped_gt_patch = cropped_gt_patch[0]
            cropped_gt_patch = final_patch

            io.imsave(os.path.join(gt_path, info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph) + ".tif"), cropped_gt_patch.astype('float32'))
            # io.imsave(os.path.join(gt_path, info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph) + "_" + str(factor) + ".tif"), cropped_gt_patch.astype('float32'))

            # input_patch_path = os.path.join(input_path, info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph))
            # os.makedirs(input_patch_path, exist_ok=True)

            # psf_bin_count = 1
            psf_count = 1
            for psf_path in psf_path_list:
                # num_psf = len(os.listdir(psf_path)) // 2
                #
                # psf_file_nums = []
                # while len(psf_file_nums) != num_psfs_per_bin:
                #     num = np.random.randint(1, num_psf + 1)
                #
                #     if num not in psf_file_nums:
                #         psf_file_nums.append(num)

                # psf_file_nums = [14, 19]
                # psf_file_nums = [11]
                # psf_file_nums = [18, 20, 4]
                psf_file_nums = [14]

                # psf_count = 0
                # psf_num_count = 1
                for psf_file_num in psf_file_nums:
                    psf_image_path = os.path.join(psf_path, str(psf_file_num) + ".tif")
                    psf_json_path = os.path.join(psf_path, str(psf_file_num) + ".json")

                    psf = io.imread(psf_image_path)

                    psf = psf / np.sum(psf)

                    convolved_patch = fftconvolution(sample=final_patch, kernel=psf)
                    # cropped_convolved_patch = crop_center([convolved_patch], sample_patch_size)
                    # cropped_convolved_patch = cropped_convolved_patch[0]
                    cropped_convolved_patch = convolved_patch
                    # convolved_noisy_patch = noise(cropped_convolved_patch)
                    convolved_noisy_patch = cropped_convolved_patch

                    psf_json_data = read_json(psf_json_path)

                    info_dict['psf_info'] = psf_json_data

                    info_dict['mean_background_offset'] = mean_background_offset
                    info_dict['sigma_background_noise'] = sigma_background_noise

                    # io.imsave(os.path.join(input_patch_path, info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph) + '-psf_' + str(psf_count) + ".tif"), convolved_noisy_patch.astype('float32'))
                    #
                    # write_json(os.path.join(input_patch_path, info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph) + '-psf_' + str(psf_count) + ".json"), info_dict)

                    io.imsave(os.path.join(input_path, info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph) + ".tif"), convolved_noisy_patch.astype('float32'))

                    write_json(os.path.join(input_path, info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph) + ".json"), info_dict)

                    # io.imsave(os.path.join(input_path, info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph) + "_" + str(factor) + ".tif"), convolved_noisy_patch.astype('float32'))
                    #
                    # write_json(os.path.join(input_path, info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph) + "_" + str(factor) + ".json"), info_dict)

                    psf_count += 1
                #     psf_num_count += 1
                # psf_bin_count += 1

            # logger.info(f"{info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph)} saved.")

        else:
            # logger.info(f"{info_dict['label'] + '-' + str(file_idx) + '-ph_' + str(ph)} discarded.")
            pass


# def main(args=None):
def main():
    start_time = time.time()
    # args = parse_args(args)
    # logger.info(args)

    dataset_name = "jrc_choroid-plexus-2"
    res_grp = "s2"
    # LABELS = ("endo-er" "mito")
    label = "mito"
    file_idx = 20
    seed = 20001
    sample_patch_size = 64
    occupancy_thresh = 0.01
    vary_intensity = 1
    num_rotations = 1
    rotation_mode = "same"
    num_psfs_per_bin = 1
    mean_background_offset = 100
    sigma_background_noise = 40

    np.random.seed(seed)

    root = r"/clusterfs_m/nvme/sayan/AI/"

    patch_dataset_root = os.path.join(root, "FIB-SEM_data_patches_overlap/", dataset_name, res_grp)

    label_list = label.split('-')
    l = len(label_list)

    # photons_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # photons_list = [400, 500, 600, 700, 800, 900, 1000]
    # photons_list = [1000]
    ph = 1000

    # for ph in photons_list:
    # for file_idx in range(5, 101, 5):
    for file_idx in range(25, 26, 5):
        label_patch_images = []
        label_patch_metadata = []

        if vary_intensity:
            vstr = "w_vary"
            for lab in label_list:
                label_patch_images.append(io.imread(os.path.join(patch_dataset_root, lab, vstr, vstr + "_" + str(ph), str(file_idx) + ".tif")))
                label_patch_metadata.append(read_json(os.path.join(patch_dataset_root, lab, vstr, vstr + "_" + str(ph), str(file_idx) + ".json")))
        else:
            vstr = "wo_vary"
            for lab in label_list:
                label_patch_images.append(io.imread(os.path.join(patch_dataset_root, lab, vstr, vstr + "_" + str(ph), str(file_idx) + ".tif")))
                label_patch_metadata.append(read_json(os.path.join(patch_dataset_root, lab, vstr, vstr + "_" + str(ph), str(file_idx) + ".json")))

        # psf_root = os.path.join(root, "psfs/psfs-ideal/")
        # psf_path_list = io.imread(os.path.join(psf_root, "ideal_norm.tif"))

        # psf_root = os.path.join(root, "psfs_updated/YuMB_lambda510/z200-y97-x97/z64-y64-x64/z15/")
        # psf_path_list = psf_file_paths(psf_root)

        psf_path_list = [os.path.join(root, "psfs_updated/YuMB_lambda510/z200-y97-x97/z64-y64-x64/z15//bimodal/photons_1-25000/amp_p14-p15/")]
        # psf_path_list = [os.path.join(root, "psfs_updated/YuMB_lambda510/z200-y97-x97/z64-y64-x64/z15/single/photons_1-25000/amp_p11-p12/")]
        # psf_path_list = [os.path.join(root, "psfs_updated/YuMB_lambda510/z200-y97-x97/z64-y64-x64/z15/single/photons_1-25000/amp_p13-p14/")]

        process_patch(dataset_name, res_grp, file_idx, label_patch_images, label_patch_metadata, label, seed, num_rotations, rotation_mode, sample_patch_size, occupancy_thresh, num_psfs_per_bin, psf_path_list, mean_background_offset, sigma_background_noise, root, ph, l)

    # logging.info(f"Total time elapsed: {time.time() - start_time:.2f} sec.")

if __name__ == "__main__":
    main()
