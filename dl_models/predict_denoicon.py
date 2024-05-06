import torch
import numpy as np
from care_v2 import CAREv2
from drunet import DRUNET, Denoicon
import os
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchio as tio # data augmentation
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage.io as io
import shutil
from tifffile import imread

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

if __name__ == '__main__':
    root = r"/clusterfs_m/nvme/sayan/AI/testing_million/psf_comp_test/"
    # root = r"/clusterfs_m/nvme/sayan/AI/testing_million/"

    exp_num = 1
    set_num = "testing_set_1"
    dataset_name = "jrc_choroid-plexus-2"
    res_grp = "s2"
    psf_num = "psf1"
    roi_num = "roi10"

    # path = os.path.join(root, set_num, dataset_name, res_grp, psf_num, roi_num)
    path = root
    # path = r"/clusterfs_m/nvme/sayan/AI/testing_big/testing_set_6_w_noise/roi2/psf2/jrc_choroid-plexus-2/s2/"

    test_path = os.path.join(path, "input/")

    ip_norm_path = os.path.join(path, "input_norm/")
    os.makedirs(ip_norm_path, exist_ok=True)

    gt_path = os.path.join(path, "gt/")


    files = []

    for file in os.listdir(test_path):
        if file.endswith(".tif"):
            files.append(file)


    # model_root = '/clusterfs_m/nvme/sayan/AI/training_big/custom_models/exp2/model4/trained/'
    model_root = '/clusterfs_m/nvme/sayan/AI/training_million/custom_models/mito_models/dnc_model2/trained/'
    # model_root = '/clusterfs_m/nvme/sayan/AI/training_million/custom_models/exp1/dru_model1/trained/'
    # model = CAREv2(in_channels=1, out_channels=1, features=[32, 64, 128])
    den = DRUNET(in_channels=1, out_channels=1, dilations=[1, 2, 4], features=[32, 64])
    dec = DRUNET(in_channels=1, out_channels=1, dilations=[1, 2, 4], features=[32, 64])
    model = Denoicon(den, dec)
    model = nn.DataParallel(model)
    model_state = torch.load(os.path.join(model_root, "saved_best_model_model1.h5"))
    model.load_state_dict(model_state['model_state_dict'])
    model.to(device)

    pred_dnc_path = os.path.join(path, "custom_predictions_mito_dnc_1/", "predictions_den_dec")
    os.makedirs(pred_dnc_path, exist_ok=True)

    pred_den_path = os.path.join(path, "custom_predictions_mito_dnc_1/", "predictions_den")
    os.makedirs(pred_den_path, exist_ok=True)


    with torch.no_grad():
        for file in files:
            x = io.imread(os.path.join(test_path, file))
            # io.imsave(os.path.join(ip_test_path, file), x)

            x -= 100
            x[x < 0] = 0
            # x = x / np.max(x)
            #
            # io.imsave(os.path.join(ip_norm_path, file), x)

            # print(np.min(x))
            # print(np.max(x))
            x = torch.from_numpy(np.array([[x]])).to(device)
            x = x.float()

            den, dec = model(x)

            den = den[0, 0].cpu().detach().numpy()
            den[den < 0] = 0
            # den = den / np.max(den)
            io.imsave(os.path.join(pred_den_path, file), den)

            dec = dec[0, 0].cpu().detach().numpy()
            dec[dec < 0] = 0
            # dec = dec / np.max(dec)
            io.imsave(os.path.join(pred_dnc_path, file), dec)

