from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
import os
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE

import sys
sys.path.append(r"../opticalaberrations/src/")

if __name__ == '__main__':
    root = r"/clusterfs_m/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psft/roi10/custom_predictions_mito_drunet_1/psf-100/"
    psf_path = os.path.join(root, "pred_psf/")

    denoised_psf_path = os.path.join(root, "denoised_pred_psf/")
    os.makedirs(denoised_psf_path, exist_ok=True)

    files = os.listdir(psf_path)

    files.sort()

    axes = 'ZYX'

    model_root = r"/clusterfs_m/fiona/Gokul/python_gu/CARE_models/"
    model = CARE(config=None, name="20231107_simulatedBeads_v3_32_64_64", basedir=model_root)
    # print(model.config)

    for file in files:
        x = imread(os.path.join(psf_path, file))

        avg_bg = np.mean(x[:,0,:]) + np.mean(x[:,-1,:]) + np.mean(x[0,:,:]) + np.mean(x[-1,:,:]) + np.mean(x[:,:,0]) + np.mean(x[:,:,-1])
        avg_bg /= 6
        x -= avg_bg
        # x -= np.percentile(x, 99)
        x[x<0] = 0

        restored = model.predict(x, axes)
        restored[restored < 0] = 0
        restored = restored / np.max(restored)

        save_file = "denoised_" + file

        save_tiff_imagej_compatible(os.path.join(denoised_psf_path, save_file), restored, axes)






