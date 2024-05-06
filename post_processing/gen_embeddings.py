import sys
sys.path.append(r"../opticalaberrations/src/")
import embeddings
import skimage.io as io
import numpy as np
import os
from scipy import stats
from skimage.filters import window

if __name__ == "__main__":
    ideal_psf = io.imread("/clusterfs_m/nvme/sayan/AI/psfs_updated/psf-ideal/YuMB_lambda510/z200-y97-x97/z64-y64-x64/z15/single/photons_5000-10000/amp_0-p0/1.tif")
    # ideal_psf = io.imread(r"/clusterfs_m/nvme/sayan/AI/psfs/psfs-ideal/ideal_norm.tif")
    ideal_otf = embeddings.fft(ideal_psf)
    ideal_otf = ideal_otf / np.nanmax(ideal_otf)

    psf_root = r"/clusterfs_m/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psft/roi10/custom_predictions_mito_dnc_1/psf-100/"
    # psf_root = r"/clusterfs_m/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/iterative_test/custom_predictions_mito_drunet_1/psf-100_5/"

    embeddings_path = os.path.join(psf_root, "embeddings/")
    os.makedirs(embeddings_path, exist_ok=True)

    gt_embeddings_path = os.path.join(embeddings_path, "gt_embeddings/")
    os.makedirs(gt_embeddings_path, exist_ok=True)

    gt_psf_path = os.path.join(psf_root, "gt_psf")
    gt_psf_files = sorted(os.listdir(gt_psf_path))

    for file in gt_psf_files:
        psf = io.imread(os.path.join(gt_psf_path, file))
        emb_path = os.path.join(gt_embeddings_path, file)
        emb = embeddings.fourier_embeddings(psf, ideal_otf, plot=emb_path)

    ip_embeddings_path = os.path.join(embeddings_path, "ip_embeddings/")
    os.makedirs(ip_embeddings_path, exist_ok=True)

    ip_psf_path = os.path.join(psf_root, "ip_psf")
    ip_psf_files = sorted(os.listdir(ip_psf_path))

    for file in ip_psf_files:
        psf = io.imread(os.path.join(ip_psf_path, file))
        emb_path = os.path.join(ip_embeddings_path, file)
        emb = embeddings.fourier_embeddings(psf, ideal_otf, plot=emb_path)

    pred_embeddings_path = os.path.join(embeddings_path, "pred_embeddings/")
    os.makedirs(pred_embeddings_path, exist_ok=True)

    pred_psf_path = os.path.join(psf_root, "pred_psf")
    pred_psf_files = sorted(os.listdir(pred_psf_path))

    for file in pred_psf_files:
        psf = io.imread(os.path.join(pred_psf_path, file))

        avg_bg = np.mean(psf[:, 0, :]) + np.mean(psf[:, -1, :]) + np.mean(psf[0, :, :]) + np.mean(psf[-1, :, :]) + np.mean(psf[:, :, 0]) + np.mean(psf[:, :, -1])
        avg_bg /= 6
        psf -= avg_bg
        psf[psf<0] = 0

        if np.isnan(psf).any():
            continue
        emb_path = os.path.join(pred_embeddings_path, file)
        emb = embeddings.fourier_embeddings(psf, ideal_otf, plot=emb_path)

    sub_mean_pred_embeddings_path = os.path.join(embeddings_path, "sub_mean_pred_embeddings/")
    os.makedirs(sub_mean_pred_embeddings_path, exist_ok=True)

    for file in pred_psf_files:
        psf = io.imread(os.path.join(pred_psf_path, file))

        psf -= np.mean(psf)
        psf[psf < 0] = 0

        if np.isnan(psf).any():
            continue
        emb_path = os.path.join(sub_mean_pred_embeddings_path, "sub_mean_" + file)
        emb = embeddings.fourier_embeddings(psf, ideal_otf, plot=emb_path)

    denoised_pred_embeddings_path = os.path.join(embeddings_path, "denoised_pred_embeddings/")
    os.makedirs(denoised_pred_embeddings_path, exist_ok=True)

    denoised_pred_psf_path = os.path.join(psf_root, "denoised_pred_psf")
    denoised_pred_psf_files = sorted(os.listdir(denoised_pred_psf_path))

    for file in denoised_pred_psf_files:
        psf = io.imread(os.path.join(denoised_pred_psf_path, file))

        if np.isnan(psf).any():
            continue
        emb_path = os.path.join(denoised_pred_embeddings_path, file)
        emb = embeddings.fourier_embeddings(psf, ideal_otf, plot=emb_path)



