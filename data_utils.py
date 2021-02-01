from os import listdir
from os.path import join

from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def getbgr(img):
    mode = img.mode
    if mode == 'RGBA':
        r, g, b, a = img.split()
    elif mode == 'RGB':
        r, g, b = img.split()
    else:
        b = [], g = [], r = []

    return r, g, b


def bgr2yuv(data_orig, bk_w, bk_h):
    R, G, B = getbgr(data_orig)
    width = data_orig.width
    height = data_orig.height

    R = np.reshape(list(R.getdata()), (height, width))
    G = np.reshape(list(G.getdata()), (height, width))
    B = np.reshape(list(B.getdata()), (height, width))

    list_yuv444 = []
    list_yuv420 = []
    list_rgb = []
    # original
    index_bk = 0
    for i in range(0, height, bk_h):
        for j in range(0, width, bk_w):

            # blocks
            # ------------------------------------------#
            # 二維
            im_new_Y_Blk = np.full((bk_h, bk_w), np.inf)
            im_new_U_Blk = np.full((bk_h, bk_w), np.inf)
            im_new_V_Blk = np.full((bk_h, bk_w), np.inf)

            im_new_R_Blk = np.full((bk_h, bk_w), np.inf)
            im_new_G_Blk = np.full((bk_h, bk_w), np.inf)
            im_new_B_Blk = np.full((bk_h, bk_w), np.inf)

            y_blk = 0
            x_blk = 0
            for y in range(i, i + bk_h):
                for x in range(j, j + bk_w):

                    # if not enough one block ... (跳掉)
                    # --------------------------#
                    if i + bk_h > height or j + bk_w > width:
                        continue
                    # --------------------------#

                    # Get value
                    # --------------------------#
                    index_all = y * bk_w + x
                    r = R[y][x]
                    g = G[y][x]
                    b = B[y][x]
                    # --------------------------#

                    # Block Setting
                    # --------------------------#
                    im_new_Y_Blk[y_blk, x_blk] = int(0.299 * r + 0.587 * g + 0.114 * b)
                    im_new_U_Blk[y_blk, x_blk] = int(-0.1687 * r - 0.3313 * g + 0.5 * b + 128)
                    im_new_V_Blk[y_blk, x_blk] = int(0.5 * r - 0.4187 * g + - 0.0813 * b + 128)
                    # --------------------------#
                    # Block rgb
                    # --------------------------#
                    im_new_R_Blk[y_blk, x_blk] = r
                    im_new_G_Blk[y_blk, x_blk] = g
                    im_new_B_Blk[y_blk, x_blk] = b
                    # --------------------------#

                    x_blk = x_blk + 1

                y_blk = y_blk + 1
                x_blk = 0
            # ------------------------------------------#

            # 420平均
            step_x = 2
            step_y = 2

            im_new_U_Blk_Ori = im_new_U_Blk.copy()
            im_new_V_Blk_Ori = im_new_V_Blk.copy()
            for y in range(0, bk_h, step_y):
                for x in range(0, bk_w, step_x):
                    # 存成一組
                    mean_U = np.mean(im_new_U_Blk[y:y + step_y, x:x + step_x])
                    mean_V = np.mean(im_new_V_Blk[y:y + step_y, x:x + step_x])

                    im_new_U_Blk[y:y + step_y, x:x + step_x].fill(mean_U)
                    im_new_V_Blk[y:y + step_y, x:x + step_x].fill(mean_V)

            # 三維
            # 422
            # Array_blk_420 = [im_new_Y_Blk, im_new_U_Blk, im_new_V_Blk]
            # yuv_420.append(Array_blk_420)
            # yuv_420 = np.append(im_new_Y_Blk,im_new_U_Blk,im_new_V_Blk,axis=0)
            yuv_444 = np.zeros(shape=(3, bk_h, bk_w))
            yuv_420 = np.zeros(shape=(3, bk_h, bk_w))
            rgb_ori = np.zeros(shape=(3, bk_h, bk_w))

            yuv_420[0] = im_new_Y_Blk
            yuv_420[1] = im_new_U_Blk
            yuv_420[2] = im_new_V_Blk
            list_yuv420.append(yuv_420)

            rgb_ori[0] = im_new_R_Blk
            rgb_ori[1] = im_new_G_Blk
            rgb_ori[2] = im_new_B_Blk
            list_rgb.append(rgb_ori)

            # 444
            # Array_blk_444 = [im_new_Y_Blk,im_new_U_Blk_Ori, im_new_V_Blk_Ori]
            # yuv_444.append(Array_blk_444)
            # yuv_444 = np.append(im_new_Y_Blk,im_new_U_Blk_Ori, im_new_V_Blk_Ori, axis=0)
            # yuv_444 = np.append(im_new_Y_Blk,im_new_U_Blk_Ori,axis=0)
            # yuv_444 = np.append(yuv_444,im_new_V_Blk_Ori,axis=0)
            yuv_444[0] = im_new_Y_Blk
            yuv_444[1] = im_new_U_Blk_Ori
            yuv_444[2] = im_new_V_Blk_Ori
            list_yuv444.append(yuv_444)

    return list_yuv444, list_yuv420, list_rgb


'''
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])
'''


def display_transform():
    return Compose([
        ToPILImage(),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, bk_width, bk_height):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.bkW = bk_width
        self.bkH = bk_height

    def __getitem__(self, index):
        bkW = self.bkW
        bkH = self.bkH
        data_orig_path = self.image_filenames[index]
        data_orig = Image.open(data_orig_path)
        data_yuv444, data_yuv420, data_rgb = bgr2yuv(data_orig, bkW, bkH)

        list_tensor_yuv444 = []
        list_tensor_yuv420 = []
        list_tensor_rgb = []
        for yuv444 in data_yuv444:
            yuv444 = (np.asarray(yuv444) / 255.0)
            yuv444_unit = torch.from_numpy(yuv444).float()
            list_tensor_yuv444.append(yuv444_unit)

        for yuv420 in data_yuv420:
            yuv420 = (np.asarray(yuv420) / 255.0)
            yuv420_unit = torch.from_numpy(yuv420).float()
            list_tensor_yuv420.append(yuv420_unit)

        for rgb in data_rgb:
            rgb = (np.asarray(rgb) / 255.0)
            rgb_unit = torch.from_numpy(rgb).float()
            list_tensor_rgb.append(rgb_unit)

        return list_tensor_yuv444, list_tensor_yuv420, list_tensor_rgb, data_orig_path

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, bk_width, bk_height):
        super(ValDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.bkW = bk_width
        self.bkH = bk_height

    def __getitem__(self, index):
        bkW = self.bkW
        bkH = self.bkH
        data_orig_path = self.image_filenames[index]
        hr_image = Image.open(data_orig_path)
        w, h = hr_image.size
        data_yuv444, data_yuv420, data_rgb = bgr2yuv(hr_image, bkW, bkH)

        list_tensor_yuv444 = []
        list_tensor_yuv420 = []
        list_tensor_rgb = []
        for yuv444 in data_yuv444:
            yuv444 = (np.asarray(yuv444) / 255.0)
            yuv444_unit = torch.from_numpy(yuv444).float()
            list_tensor_yuv444.append(yuv444_unit)

        for yuv420 in data_yuv420:
            yuv420 = (np.asarray(yuv420) / 255.0)
            yuv420_unit = torch.from_numpy(yuv420).float()
            list_tensor_yuv420.append(yuv420_unit)

        for rgb in data_rgb:
            rgb = (np.asarray(rgb) / 255.0)
            rgb_unit = torch.from_numpy(rgb).float()
            list_tensor_rgb.append(rgb_unit)

        return list_tensor_yuv444, list_tensor_yuv420, list_tensor_rgb, data_orig_path

    def __len__(self):
        return len(self.image_filenames)


'''
class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
'''
