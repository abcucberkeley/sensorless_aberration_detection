import torch
import numpy as np
from drunet import DRUNET
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
    # root = r"/clusterfs_m/nvme/sayan/AI/testing_million/testing_set_1/jrc_choroid-plexus-2/s2/psf1/roi14/"
    root = r"/clusterfs_m/nvme/sayan/AI/testing_128/psf_comp_test_125/mode_14/"

    exp_num = 1
    # set_num = "testing_set_" + str(exp_num) + "_wo_noise"
    # set_num = "testing_set_5_w_noise"
    set_num = "testing_set_1"
    # set_num = "training_set_2"
    # set_num = os.path.join("testing_set_6_w_noise", "roi3", "psf4")
    dataset_name = "jrc_choroid-plexus-2"
    res_grp = "s2"
    psf_num = "psf6"
    roi_num = "roi10"

    # path = os.path.join(root, set_num, dataset_name, res_grp, psf_num, roi_num)
    path = root
    # path = r"/clusterfs_m/nvme/sayan/AI/testing_big/testing_set_6_w_noise/roi2/psf2/jrc_choroid-plexus-2/s2/"

    test_path = os.path.join(path, "input/")

    # ip_norm_path = os.path.join(path, "input_norm/")
    # os.makedirs(ip_norm_path, exist_ok=True)

    gt_path = os.path.join(path, "gt/")

    # gt_norm_path = os.path.join(path, "gt_norm/")
    # os.makedirs(gt_norm_path, exist_ok=True)

    files = []
    # c = 0
    for file in os.listdir(test_path):
        if file.endswith(".tif"):
            files.append(file)
        #     c += 1
        # if c >= 30:
        #     break

    # files.sort()

    # model_root = '/clusterfs_m/nvme/sayan/AI/training_big/custom_models/exp2/model4/trained/'
    # model_root = '/clusterfs_m/nvme/sayan/AI/training_million/custom_models/mito_models/dru_model4/trained/'
    model_root = '/clusterfs_m/nvme/sayan/AI/training_128/custom_models/mito_models/dru128_model1/trained/'
    # model_root = '/clusterfs_m/nvme/sayan/AI/training_million/custom_models/mito_models/dru_denoise_2_model1/trained/'
    # model_root = '/clusterfs_m/nvme/sayan/AI/training_million/custom_models/mito_models/dru_densuboff_to_deconv_model1/trained/'
    # model = DRUNET(in_channels=1, out_channels=1, dilations=[1, 2, 4], features=[32, 64])
    model = DRUNET(in_channels=1, out_channels=1, dropout=0.02, dilations=[1, 2, 4, 8], features=[32, 64, 128])
    model = nn.DataParallel(model)
    model_state = torch.load(os.path.join(model_root, "saved_best_model_model1.h5"))
    model.load_state_dict(model_state['model_state_dict'])
    model.to(device)

    # ip_test_path = os.path.join(path, "input-test/")
    # os.makedirs(ip_test_path, exist_ok=True)
    #
    # gt_test_path = os.path.join(path, "gt-test/")
    # os.makedirs(gt_test_path, exist_ok=True)

    pred_path = os.path.join(path, "custom_predictions_mito_drunet128_4/", "predictions")
    # pred_path = os.path.join(path, "custom_predictions_mito_denoise_drunet_1/", "predictions")
    # pred_path = os.path.join(path, "custom_predictions_mito_densuboff_to_deconv_drunet_1/", "predictions")
    os.makedirs(pred_path, exist_ok=True)

    # abs_diff_path = os.path.join(path, "custom_predictions_mito_drunet_4/", "abs_diff")
    # os.makedirs(abs_diff_path, exist_ok=True)
    # #
    # # # ideal_psf_path = r"/clusterfs/nvme/sayan/AI/psfs/psfs-ideal/ideal_norm.tif"
    # # # ideal_psf = imread(ideal_psf_path)
    # # # ideal_psf /= np.sum(ideal_psf)
    # #
    # # # binarized_path = os.path.join(path, "binarized/")
    # # # Path(binarized_path).mkdir(exist_ok=True)
    # #
    with torch.no_grad():
        model.eval()

        for file in files:
            x = io.imread(os.path.join(test_path, file))
            # io.imsave(os.path.join(ip_test_path, file), x)

            # y = io.imread(os.path.join(gt_path, file))

            # json_file = file[:-3] + "json"
            # shutil.copy2(os.path.join(test_path, json_file), os.path.join(ip_test_path, json_file))

            x -= 100
            x[x < 0] = 0
            # x = x / np.max(x)
            #
            # io.imsave(os.path.join(ip_norm_path, file), x)

            # print(np.min(x))
            # print(np.max(x))
            x = torch.from_numpy(np.array([[x]])).to(device)
            x = x.float()

            restored = model(x)
            restored = restored[0, 0].cpu().detach().numpy()
            restored[restored < 0] = 0
            # restored = restored / np.max(restored)
            # restored *= np.max(y)
            io.imsave(os.path.join(pred_path, file), restored)

            # print(restored.shape)

            # print(np.min(restored))
            # print(np.max(restored))

            # file_split = file.split("-")
            # file_split[-1] = file_split[-1][-4:]
            # gt_file = file_split[0]
            # for s in file_split[1:-1]:
            #     gt_file += "-" + s
            # gt_file += file_split[-1]

            # y = io.imread(os.path.join(gt_path, gt_file))
            # y = io.imread(os.path.join(gt_path, file))

            # io.imsave(os.path.join(gt_test_path, file), y)

            # y = y / np.max(y)
            # io.imsave(os.path.join(gt_norm_path, file), y)
            # y = y / np.sum(y)

            # abs_diff = abs(y - restored)
            # io.imsave(os.path.join(abs_diff_path, "abs_diff-" + file), abs_diff)
