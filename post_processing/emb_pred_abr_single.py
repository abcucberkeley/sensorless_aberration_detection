import sys
sys.path.append(r"../opticalaberrations/src/")
import experimental
from pathlib import Path
import os
import shutil
import time
import logging
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import cli


# def main(args=None):
def main():
    # psf_path = "/clusterfs_m/nvme/sayan/AI/testing_million/tt/"
    # file = "psf_1.tif"
    model_name = "mito"
    # model_name = "exp1"
    # model_name = "er"
    psf_num = 6
    roi_num = 10
    psf_root = "/clusterfs_m/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf" + str(psf_num) + "/roi" + str(roi_num) + "/custom_predictions_" + model_name + "_drunet_4/psf-100/"
    psf_root = "/clusterfs_m/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psft/roi10/custom_predictions_mito_drunet_1/psf-100/"
    # psf_filenames = ["gt_psf", "denoised_pred_psf"]


    model_path = "../opticalaberrations/pretrained_models/opticalnet-15-YuMB-lambda510.h5"
    dm_cal = "../opticalaberrations/calibration/aang/15_mode_calibration.csv"

    psf_file = "denoised_pred_psf"
    # psf_file = "gt_psf"
    psf_path = os.path.join(psf_root, psf_file)
    filenames = []
    for file in os.listdir(psf_path):
        filenames.append(file)
        experimental.predict_sample(Path(psf_path) / file, Path(model_path), Path(dm_cal), None, axial_voxel_size=.200, lateral_voxel_size=.097, wavelength=.510, batch_size=896, plot=True)

    int_path_str = ["sample_predictions_corrected_actuators.csv", "sample_predictions_diagnosis.svg", "sample_predictions_embeddings.png", "sample_predictions_embeddings.svg", "sample_predictions_preprocessing.svg", "sample_predictions_psf.tif", "sample_predictions_rotations.csv", "sample_predictions_settings.json", "sample_predictions_wavefront.tif", "sample_predictions_zernike_coefficients.csv"]

    int_paths = []
    for i_p_s in int_path_str:
        i_path = os.path.join(psf_root, psf_file + "_inference", i_p_s.split('.')[0] + "_" + i_p_s.split('.')[1])
        os.makedirs(i_path, exist_ok=True)
        int_paths.append(i_path)

    for file in filenames:
        file = file.split('.')[0]
        for i in range(len(int_paths)):
            shutil.move(os.path.join(psf_path, file + "_" + int_path_str[i]), os.path.join(int_paths[i], file + "_" + int_path_str[i]))


if __name__ == "__main__":
    main()
