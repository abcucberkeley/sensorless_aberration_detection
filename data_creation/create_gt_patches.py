import sys
import logging
sys.path.append(r"../opticalaberrations/src/")
import cli

import os
import skimage.io as io
import numpy as np
from fibsem_tools import read_xarray
from skimage import util
from skimage.transform import resize
import random
import time
import json
from skimage.morphology import dilation, ball

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def patch_indices(sample, shape=(168, 168, 168), step_size=168):
    patch_info = []
    x, y, z = 0, 0, 1
    while z + shape[0] < sample.shape[0]:
        while y + shape[1] < sample.shape[1]:
            while x + shape[2] < sample.shape[2]:
                patch_info.append((str(z)+':'+str(z+shape[0]), str(y)+':'+str(y+shape[1]), str(x)+':'+str(x+shape[2])))
                x += step_size
            x = 0
            y += step_size
        y = 0
        z += step_size

    return patch_info


def write_json(path, data):
    file = open(path, 'w')
    json.dump(data, file, indent=4)
    file.close()


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("--dataset_name", type=str, default="jrc_choroid-plexus-2", help="Name of the dataset")

    parser.add_argument("--res_grp", type=str, default="s2", help="Resolution group")

    parser.add_argument("--label", type=str, default="mito", help="Name of the label")

    parser.add_argument("--seed", type=int, default=10000, help="Random seed")

    parser.add_argument("--photons", type=int, default=1000, help="Photons")

    parser.add_argument("--aug_patch_size", type=int, default=168, help="Bigger patch size needed for augmentations")

    parser.add_argument("--step_size", type=int, default=168, help="Determine overlap")

    parser.add_argument("--vary_intensity", type=int, default=0, help="Determine whether to vary Intensity")

    return parser.parse_args(args)


def main(args=None):
    start_time = time.time()
    args = parse_args(args)
    logger.info(args)

    dataset_name = args.dataset_name
    res_grp = args.res_grp
    label = args.label
    seed = args.seed
    photons = args.photons
    aug_patch_size = args.aug_patch_size
    step_size = args.step_size
    vary_intensity = args.vary_intensity

    np.random.seed(seed)

    dimension_dict = {"s0": 8.0, "s1": 16.0, "s2": 32.0, "s3": 64.0, "s4": 128.0, "s5": 256.0}

    root = r"/clusterfs/nvme/sayan/AI/"

    dataset_root = os.path.join(root, "FIB-SEM_data_n5/", dataset_name, dataset_name + ".n5", "volumes/")

    raw_path = os.path.join(dataset_root, "raw/", res_grp)

    info_dict = {}

    info_dict['dataset_name'] = dataset_name
    info_dict['resolution_group'] = res_grp
    info_dict['order'] = "c"
    info_dict['voxel_resolution'] = (dimension_dict[res_grp], dimension_dict[res_grp], dimension_dict[res_grp])
    info_dict['label'] = label

    raw_image = read_xarray(raw_path).compute().data
    raw_image = util.invert(raw_image)

    logger.info(f"Raw file loaded and inverted.")

    info_dict['dimensions'] = raw_image.shape

    info_dict['augmentation_patch_size'] = (aug_patch_size, aug_patch_size, aug_patch_size)

    info_dict['photons'] = photons

    info_dict['seed'] = seed

    info_dict['step_size'] = step_size

    label_path = os.path.join(dataset_root, "labels/", label + "_seg/", res_grp)
    label_image = read_xarray(label_path).compute().data

    logger.info(f"Label file loaded.")

    unique_ids = np.unique(label_image)

    logger.info(f"# unique labels: {len(unique_ids)}.")

    if vary_intensity:
        intensity_scale_dict = {0: 0}
        for id in unique_ids[1:]:
            scale = np.random.normal(1, .333)
            intensity_scale_dict[id] = scale if scale > 0 else 0

        for i in range(label_image.shape[0]):
            for j in range(label_image.shape[1]):
                for k in range(label_image.shape[2]):
                    label_image[i, j, k] = intensity_scale_dict[label_image[i, j, k]] * photons

    else:
        label_image[label_image > 0] = photons

    label_image = resize(label_image, (label_image.shape[0] // 2, label_image.shape[1] // 2, label_image.shape[2] // 2), order=3, anti_aliasing=True, preserve_range=True)

    logger.info(f"Label file updated.")

    patch_info = patch_indices(raw_image, shape=(aug_patch_size, aug_patch_size, aug_patch_size), step_size=step_size)

    logger.info(f"Patch indices created.")

    if vary_intensity:
        save_path = os.path.join(root, "FIB-SEM_data_patches_375/", dataset_name, res_grp, label, "w_vary", "w_vary_" + str(photons))
    else:
        save_path = os.path.join(root, "FIB-SEM_data_patches_375/", dataset_name, res_grp, label, "wo_vary", "wo_vary_" + str(photons))
    os.makedirs(save_path, exist_ok=True)

    io.imsave(os.path.join(save_path, label + ".tif"), label_image.astype('uint16'))

    for i in range(len(patch_info)):
        z_range = patch_info[i][0].split(sep=':')
        y_range = patch_info[i][1].split(sep=':')
        x_range = patch_info[i][2].split(sep=':')

        raw_image_patch = raw_image[int(z_range[0]):int(z_range[1]), int(y_range[0]):int(y_range[1]), int(x_range[0]):int(x_range[1])]
        raw_image_patch = raw_image_patch / np.max(raw_image_patch)
        label_image_patch = label_image[int(z_range[0]):int(z_range[1]), int(y_range[0]):int(y_range[1]), int(x_range[0]):int(x_range[1])]

        raw_x_label_image_patch = raw_image_patch * label_image_patch

        info_dict['patch_info'] = patch_info[i]

        if vary_intensity:
            info_dict['label_intensity_variation_function'] = "np.random.normal(1, .33)"
        else:
            info_dict['label_intensity_variation_function'] = "none"

        io.imsave(os.path.join(save_path, str(i+1) + ".tif"), raw_x_label_image_patch.astype('float32'))
        write_json(os.path.join(save_path, str(i+1) + ".json"), info_dict)

        logger.info(f"File {i+1} out of {len(patch_info)} saved.")

    logging.info(f"Total time elapsed: {time.time() - start_time:.2f} sec.")

if __name__ == "__main__":
    main()


