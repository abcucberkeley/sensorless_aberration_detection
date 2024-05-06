import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from drunet import DRUNET, Denoicon
from dataloader import data_loader
# from dataloader_denoicon import data_loader
import json
import os
import sys
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchio as tio # data augmentation
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import cupy as cp
from skimage.filters import window, difference_of_gaussians

import logging
import time

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# import sys
sys.path.append(r"/global/home/users/sayanseal/git-managed/opticalaberrations/src/")
# # sys.path.append(r"/home/sayanseal/git-managed/opticalaberrations/src/")
#
# from preprocessing import dog

def dog(image, low_sigma=0.7, high_sigma=3):
    # logger.info(image.size())
    filtered_image = torch.zeros_like(image)
    gaussian_blur_low = T.GaussianBlur(kernel_size=5, sigma=low_sigma)

    gaussian_blur_high = T.GaussianBlur(kernel_size=5, sigma=high_sigma)

    # with torch.no_grad():
    for b in range(len(image)):
        filtered_image[b, 0, :, :, :] = gaussian_blur_low(image[b, 0, :, :, :]) - gaussian_blur_high(image[b, 0, :, :, :])

    filtered_image[filtered_image < 0] = 0

    return filtered_image

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_json(path):
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    return data


if __name__ == '__main__':
    start_time = time.time()
    pmem = True if device == "cuda" else False

    exp_num = 1
    model_num = 1
    root = r"/clusterfs/nvme/sayan/AI/training_128/"

    # model_root = os.path.join(root, "custom_models", "exp" + str(exp_num), "dru_model" + str(model_num))
    model_root = os.path.join(root, "custom_models", "mito_models", "dru128_1000_model" + str(model_num))
    os.makedirs(model_root, exist_ok=True)

    writer = SummaryWriter(os.path.join(model_root, "tensorboard_logs"))

    config = read_json("config.json")

    dataset_name = "jrc_choroid-plexus-2"
    res_grp = "s2"

    # train_root = os.path.join(root, "training_set_" + str(exp_num), dataset_name, res_grp)
    train_root = os.path.join(root, "mito_training_set_1000", dataset_name, res_grp)

    input_path = os.path.join(train_root, "input/")
    # input_path = os.path.join(train_root, "convolved/")
    gt_path = os.path.join(train_root, "gt/")

    # input_path = os.path.join(train_root, "denoised_sub_offset/")
    # gt_path = os.path.join(train_root, "gt_1000/")
    validation_split_pc = .2

    train_dataloader, validation_dataloader = data_loader(input_path, gt_path, config['batch_size'], pmem, validation_split_pc)

    # model = DRUNET(in_channels=1, out_channels=1, dilations=[1, 2, 4], features=[16, 16])
    model = DRUNET(in_channels=1, out_channels=1, dropout=0.02, dilations=[1, 2, 4, 8], features=[32, 64, 128])
    # model = DRUNET(in_channels=1, out_channels=1, dilations=[1, 2, 4], features=[32, 64])
    # den = DRUNET(in_channels=1, out_channels=1, dilations=[1, 2, 4], features=[32, 64])
    # dec = DRUNET(in_channels=1, out_channels=1, dilations=[1, 2, 4], features=[32, 64])
    # model = Denoicon(den, dec)
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model = nn.DataParallel(model)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[0, 1, 2, 3])
    # summary(model)
    model.to(device)

    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    checkpoint_save_path = os.path.join(model_root, "checkpoints")
    os.makedirs(checkpoint_save_path, exist_ok=True)

    model_save_path = os.path.join(model_root, "trained")
    os.makedirs(model_save_path, exist_ok=True)

    plot_save_path = os.path.join(model_root, "plots")
    os.makedirs(plot_save_path, exist_ok=True)

    hist = {"train_loss": [], "validation_loss": []}

    min_validation_loss = 10000

    for e in tqdm(range(config['epochs'])):
        model.train()

        total_train_loss = 0
        total_validation_loss = 0

        for (i, data) in enumerate(train_dataloader):
            (x, y) = (data[0].cuda(), data[1].cuda())
            # (x, d, y) = (data[0].cuda(), data[1].cuda(), data[2].cuda())

            pred = model(x)
            # den, dec = model(x)

            loss = criterion(pred, y)

            # logger.info("Loss grad: {}".format(loss.requires_grad))
            # loss_den = criterion(den, d)
            # loss_dec = criterion(dec, y)
            # loss = loss_den + loss_dec

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss

        with torch.no_grad():
            model.eval()

            for (i, data) in enumerate(validation_dataloader):
                (x, y) = (data[0].to(device), data[1].to(device))
                pred = model(x)
                total_validation_loss += criterion(pred, y)
                # (x, d, y) = (data[0].to(device), data[1].to(device), data[2].to(device))
                # den, dec = model(x)
                # total_validation_loss += criterion(den, d) + criterion(dec, y)

        mean_train_loss = total_train_loss / len(train_dataloader.dataset)
        mean_validation_loss = total_validation_loss / len(validation_dataloader.dataset)

        writer.add_scalars("loss", {'train': mean_train_loss, 'validation': mean_validation_loss}, e)

        scheduler.step(mean_validation_loss)

        hist["train_loss"].append(mean_train_loss.cpu().detach().numpy())
        hist["validation_loss"].append(mean_validation_loss.cpu().detach().numpy())

        logging.info("[INFO] EPOCH: {}/{}".format(e+1, config['epochs']))
        logging.info("Train loss: {:.6f}, Validation loss: {:.6f}".format(mean_train_loss, mean_validation_loss))

        if e % 10 == 0:
            torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': mean_train_loss,
                        'validation_loss': mean_validation_loss
                        },
                os.path.join(checkpoint_save_path, config['name'] + f'_checkpoint_{e}.h5'))

        if mean_validation_loss < min_validation_loss:
            min_validation_loss = mean_validation_loss
            torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': mean_train_loss,
                        'validation_loss': mean_validation_loss
                        },
                os.path.join(model_save_path, "saved_best_model_" + config['name'] + '.h5'))

    torch.save({'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': scheduler.state_dict(),
                'train_loss': mean_train_loss,
                'validation_loss': mean_validation_loss
               },
               os.path.join(model_save_path, "saved_last_model_" + config['name'] + '.h5'))

    writer.close()

    logging.info(f"Total time elapsed: {time.time() - start_time:.2f} sec.")

    plt.figure()
    plt.plot(hist["train_loss"], label="train_loss")
    plt.plot(hist["validation_loss"], label="validation_loss")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join(plot_save_path, "loss_" + config['name'] + '.png'))
