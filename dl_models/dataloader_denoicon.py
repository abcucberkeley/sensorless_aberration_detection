import torch
import os
import numpy as np
import skimage.io as io
import time


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_files, den_files, gt_files):
        self.input_files = input_files
        self.den_files = den_files
        self.gt_files = gt_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        x = io.imread(self.input_files[idx])
        den = io.imread(self.den_files[idx])
        y = io.imread(self.gt_files[idx])

        x -= 100
        x[x < 0] = 0

        # x = x/np.max(x)
        # den = den/np.max(den)
        # y = y/np.max(y)

        # y *= .82/.22
        # y[y < 0] = 0

        return np.array([x]), np.array([den]), np.array([y])


def data_loader(input_path, den_path, gt_path, bsize, pmem, validation_split_pc=0.2):
    # # gt_filenames = sorted(os.listdir(gt_path), key=lambda x: (x.rsplit('-', 2)[0], int(x.rsplit('-', 2)[1]), int(x.rsplit('-', 2)[2].split('_')[1].split('.')[0]), x.rsplit('-', 2)[2].split('_')[1].split('.')[1]))
    # gt_filenames = sorted(os.listdir(gt_path), key=lambda x: (x.rsplit('-', 2)[0], int(x.rsplit('-', 2)[1]), int(x.rsplit('-', 2)[2].split('_')[1].split('.')[0]), x.rsplit('-', 2)[2].split('_')[1].split('.')[1]))
    # gt_filenames = np.repeat(gt_filenames, 60)
    # # gt_filenames = np.array(sorted(os.listdir(gt_path), key=lambda x: (x.rsplit('-', 3)[0], int(x.rsplit('-', 3)[1]), int(x.rsplit('-', 3)[2].split('_')[1]), int(x.rsplit('-', 3)[3].split('_')[1].split('.')[0]), x.rsplit('-', 3)[3].split('_')[1].split('.')[1])))
    # input_all_filenames = np.array(sorted(os.listdir(input_path), key=lambda x: (x.rsplit('-', 3)[0], int(x.rsplit('-', 3)[1]), int(x.rsplit('-', 3)[2].split('_')[1]), int(x.rsplit('-', 3)[3].split('_')[1].split('.')[0]), x.rsplit('-', 3)[3].split('_')[1].split('.')[1])))
    #
    # # gt_filenames = np.array(sorted(os.listdir(gt_path), key=lambda x: (x.rsplit('-', 1)[0], int(x.rsplit('-', 1)[1].split('.')[0]), x.rsplit('-', 1)[1].split('.')[1])))
    #
    # num_files = len(gt_filenames)
    #
    # input_filenames = []
    # for file in input_all_filenames:
    #     if file.endswith('.tif'):
    #         input_filenames.append(file)
    # input_filenames = np.array(input_filenames)
    # # input_filenames = gt_filenames

    gt_filenames = sorted(os.listdir(gt_path), key=lambda x: (x.rsplit('-', 2)[0], int(x.rsplit('-', 2)[1]), int(x.rsplit('-', 2)[2].split('_')[1].split('.')[0])))
    gt_filenames = np.repeat(gt_filenames, 300)
    input_filepaths = sorted(os.listdir(input_path), key=lambda x: (x.rsplit('-', 2)[0], int(x.rsplit('-', 2)[1]), int(x.rsplit('-', 2)[2].split('_')[1].split('.')[0])))
    den_filepaths = sorted(os.listdir(den_path), key=lambda x: (x.rsplit('-', 2)[0], int(x.rsplit('-', 2)[1]), int(x.rsplit('-', 2)[2].split('_')[1].split('.')[0])))

    input_all_filenames = []
    for file in input_filepaths:
        sorted_input_files = sorted(os.listdir(os.path.join(input_path, file)), key=lambda x: (x.rsplit('-', 3)[0], int(x.rsplit('-', 3)[1]), int(x.rsplit('-', 3)[2].split('_')[1]), int(x.rsplit('-', 3)[3].split('_')[1].split('.')[0]), x.rsplit('-', 3)[3].split('_')[1].split('.')[1]))
        sorted_input_filepaths = []
        for i in range(len(sorted_input_files)):
            sorted_input_filepaths.append(os.path.join(file, sorted_input_files[i]))
        input_all_filenames.extend(np.array(sorted_input_filepaths))

    num_files = len(gt_filenames)

    input_filenames = []
    for file in input_all_filenames:
        if file.endswith('.tif'):
            input_filenames.append(file)
    input_filenames = np.array(input_filenames)

    den_filenames = []
    for file in den_filepaths:
        sorted_den_files = sorted(os.listdir(os.path.join(den_path, file)), key=lambda x: (x.rsplit('-', 3)[0], int(x.rsplit('-', 3)[1]), int(x.rsplit('-', 3)[2].split('_')[1]), int(x.rsplit('-', 3)[3].split('_')[1].split('.')[0]), x.rsplit('-', 3)[3].split('_')[1].split('.')[1]))
        sorted_den_filepaths = []
        for i in range(len(sorted_den_files)):
            sorted_den_filepaths.append(os.path.join(file, sorted_den_files[i]))
        den_filenames.extend(np.array(sorted_den_filepaths))
    den_filenames = np.array(den_filenames)

    idx = np.arange(num_files)
    np.random.shuffle(idx)

    nvalidation = int(num_files * validation_split_pc)
    ntrain = num_files - nvalidation

    train_idx = idx[:ntrain]
    train_gt_filenames = gt_filenames[train_idx]
    train_input_filenames = input_filenames[train_idx]
    train_den_filenames = den_filenames[train_idx]
    print('Training size: ', len(train_gt_filenames))

    train_gt_files = []
    train_input_files = []
    train_den_files = []
    for i in range(ntrain):
        train_gt_files.append(os.path.join(gt_path, train_gt_filenames[i]))
        train_input_files.append(os.path.join(input_path, train_input_filenames[i]))
        train_den_files.append(os.path.join(den_path, train_den_filenames[i]))

    train_data = Dataset(train_input_files, train_den_files, train_gt_files)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bsize, shuffle=True, pin_memory=pmem)

    if validation_split_pc > 0:
        validation_idx = idx[ntrain:]
        validation_gt_filenames = gt_filenames[validation_idx]
        validation_input_filenames = input_filenames[validation_idx]
        validation_den_filenames = den_filenames[validation_idx]
        print('Validation size: ', len(validation_gt_filenames))

        validation_gt_files = []
        validation_input_files = []
        validation_den_files = []
        for i in range(nvalidation):
            validation_gt_files.append(os.path.join(gt_path, validation_gt_filenames[i]))
            validation_input_files.append(os.path.join(input_path, validation_input_filenames[i]))
            validation_den_files.append(os.path.join(den_path, validation_den_filenames[i]))

    else:
        validation_gt_files = train_gt_files.copy()
        print('Validation size: ', len(validation_gt_files))
        validation_input_files = train_input_files.copy()
        validation_den_files = train_den_files.copy()

    validation_data = Dataset(validation_input_files, validation_den_files, validation_gt_files)
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=bsize, pin_memory=pmem)

    return train_dataloader, validation_dataloader


# def main(args=None):
if __name__ == '__main__':
    pass
