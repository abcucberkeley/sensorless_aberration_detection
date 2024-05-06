import numpy as np
from fibsem_tools import read_xarray
import skimage.io as io
from skimage.transform import rescale, resize
from skimage import util
import time
import os
import sys
import logging
sys.path.append(r"../opticalaberrations/src/")
import cli

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_raw(raw_path, save_path, res_grp):
    raw_image = read_xarray(raw_path)

    io.imsave(os.path.join(save_path, "raw-" + res_grp + ".tif"), raw_image)
    io.imsave(os.path.join(save_path, "raw-" + res_grp + "-inverted.tif"), util.invert(raw_image))

    logger.info(f"Raw Image saved: {save_path}")


def read_pred(pred_path, save_path, res_grp):
    pred_image = read_xarray(pred_path)

    #pred_image_resized = resize(pred_image, (pred_image.shape[0] // 2, pred_image.shape[1] // 2, pred_image.shape[2] // 2), order=3, anti_aliasing=True)  # loads into memory

    io.imsave(os.path.join(save_path, pred_path[114:-3] + "-" + res_grp + ".tif"), pred_image)

    logger.info(f"{pred_path[114:-3]} mask saved: {save_path}")

def read_seg(seg_path, save_path, res_grp):
    seg_image = read_xarray(seg_path)

    #seg_image_resized = resize(seg_image, (seg_image.shape[0] // 2, seg_image.shape[1] // 2, seg_image.shape[2] // 2), order=3, anti_aliasing=True)  # loads into memory

    io.imsave(os.path.join(save_path, seg_path[114:-3] + "-" + res_grp + ".tif"), seg_image)

    logger.info(f"{seg_path[114:-3]} mask saved: {save_path}")


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("--dataset_name", type=str, default="jrc_choroid-plexus-2", help="Name of the dataset")

    parser.add_argument("--res_grp", type=str, default="s2", help="Resolution group")

    return parser.parse_args(args)


def main(args=None):
    start_time = time.time()
    args = parse_args(args)
    logger.info(args)

    dataset_name = args.dataset_name
    res_grp = args.res_grp
    save_root = r"/clusterfs/nvme/sayan/AI/FIB-SEM_data/"
    save_path = os.path.join(save_root, dataset_name, res_grp)
    # os.makedirs(save_path, exist_ok=True)

    dataset_root = os.path.join(r"/clusterfs/abc/ABCData/20230224-FIBSEM-heinrich-2021a/", dataset_name, dataset_name + ".n5", "volumes/")

    raw_path = os.path.join(dataset_root, "raw/", res_grp)
    read_raw(raw_path, save_path, res_grp)

    pred_paths = []
    seg_paths = []
    labels_root = os.path.join(dataset_root, "labels/")

    for dir in os.listdir(labels_root):
        if "pred" in dir:
            pred_paths.append(os.path.join(labels_root, dir, res_grp))
        elif "seg" in dir:
            seg_paths.append(os.path.join(labels_root, dir, res_grp))

    pred_save_path = os.path.join(save_path, "pred/")
    os.makedirs(pred_save_path, exist_ok=True)
    
    seg_save_path = os.path.join(save_path, "seg/")
    os.makedirs(seg_save_path, exist_ok=True)

    for pred_path in pred_paths:
        read_pred(pred_path, pred_save_path, res_grp)

    for seg_path in seg_paths:
        read_seg(seg_path, seg_save_path, res_grp)

    logging.info(f"Total time elapsed: {time.time() - start_time:.2f} sec.")


if __name__ == "__main__":
    main()
