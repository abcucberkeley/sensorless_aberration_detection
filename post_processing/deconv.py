import os
import numpy as np
import skimage.io as io
import json
import tifffile

from skimage import restoration
import skimage.filters as fil
from aydin.restoration.denoise.noise2selfcnn import noise2self_cnn, Noise2SelfCNN

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


def read_json(path):
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    return data


# def wiener_deconvolution(signal, kernel, lambd):
#     # kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel)))) # zero pad the kernel to same length
#     H = fft(kernel)
#     deconvolved = np.real(ifft(fft(signal)*np.conj(H)/(H*np.conj(H) + lambd**2)))
#     return deconvolved

()
def wiener_deconvolution(signal, kernel, snr):
    H = fft(kernel)
    deconvolved = np.real(ifft(((H*np.conj(H))/(H*np.conj(H) + 1/snr)) * (fft(signal)/H)))
    return deconvolved


if __name__ == "__main__":
    ip_image = io.imread(r"/clusterfs_m/nvme/sayan/AI/testing_million/psf_comp_test/input/mito-45-ph_1000.tif")
    psf = io.imread(r"/clusterfs_m/nvme/sayan/AI/testing_million/psf_comp_test/gt/mito-45-ph_1000.tif")

    # decon = restoration.wiener(ip_image, psf, 0.01)
    decon = wiener_deconvolution(ip_image, psf, 30)
    decon[decon < 0] = 0
    # decon /= np.max(decon)
    #
    # io.imsave("/clusterfs_m/nvme/sayan/AI/testing_v2/testing_set_6_emb_2/jrc_choroid-plexus-2/s2/ip_deconv_endo-er-36-ph_1000-psf2.tif", decon)

    # decon = restoration.richardson_lucy(ip_image, psf, num_iter=50)
    # decon = restoration.wiener(ip_image, psf, 0.5)
    # decon[decon < 0] = 0
    # decon /= np.max(decon)
    # io.imsave(r"/clusterfs_m/nvme/sayan/AI/testing_million/psf_comp_test/gt/mito-45-ph_1000_psf.tif", decon)

    # n2s = Noise2SelfCNN()
    # n2s.train(np.array(ip_image))
    # denoised_image = n2s.denoise(np.array(ip_image))
    # denoised_image = noise2self_cnn(r"/clusterfs_m/nvme/sayan/AI/testing_million/psf_comp_test/input/mito-45-ph_1000.tif")

    # io.imsave(r"/clusterfs_m/nvme/sayan/AI/testing_million/psf_comp_test/gt/mito-45-ph_1000_den.tif", denoised_image)

    # img = fil.gaussian(ip_image, sigma=1)
    io.imsave("/clusterfs_m/nvme/sayan/AI/testing_million/psf_comp_test/input/mito-45-ph_1000_wiener.tif", decon.astype(np.float32))













