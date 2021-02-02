import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
# parser.add_argument('--upscale_factor', default=1, type=int, choices=[1, 2, 4, 8],
#                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--use_cuda', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--block_width', type=int, default=32)
parser.add_argument('--block_height', type=int, default=32)
parser.add_argument('--train_path', type=str, default='')
parser.add_argument('--valid_path', type=str, default='')
parser.add_argument('--statistics_path', type=str, default='')
parser.add_argument('--epochs_path', type=str, default='')
parser.add_argument('--snapshots_folder', type=str, default='')
parser.add_argument('--snapshots_train_data', type=str, default='')


'''
statistics

epochs
'''

if __name__ == '__main__':
    opt = parser.parse_args()

    # CROP_SIZE = opt.crop_size
    # UPSCALE_FACTOR = opt.upscale_factor
    BK_WIDTH = opt.block_width
    BK_HEIGHT = opt.block_height
    NUM_EPOCHS = opt.num_epochs
    USE_CUDA = opt.use_cuda
    BATCH_SIZE = opt.batch_size
    TRAIN_RESULT_FOLDER = opt.snapshots_folder
    TRAIN_RESULT_PARAMETERS = opt.snapshots_train_data
    TRAIN_PATH = opt.train_path
    VALID_PATH = opt.valid_path

    # train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # train_set = TrainDatasetFromFolder('data/DIV2K_test_index', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)
    train_set = TrainDatasetFromFolder(TRAIN_PATH, BK_WIDTH, BK_HEIGHT)
    val_set = ValDatasetFromFolder(VALID_PATH, BK_WIDTH, BK_HEIGHT)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator()
    if TRAIN_RESULT_FOLDER:
        if USE_CUDA == 1 and torch.cuda.is_available():
            netG.load_state_dict(torch.load(TRAIN_RESULT_FOLDER + TRAIN_RESULT_PARAMETERS))
        else:
            netG.load_state_dict(torch.load(TRAIN_RESULT_FOLDER + TRAIN_RESULT_PARAMETERS, map_location='cpu'))

    else:
        print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()

    if USE_CUDA == 1 and torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        print('before train bar')
        for datas, targets, rgbs, path in train_bar:

            for index in range(len(datas)):
                data = datas[index]
                target = targets[index]
                index_size = len(datas)
                g_update_first = True
                batch_size = data.size(0)
                running_results['batch_sizes'] += batch_size

                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                real_img = Variable(target)
                if USE_CUDA == 1 and torch.cuda.is_available():
                    real_img = real_img.cuda()
                z = Variable(data)
                if USE_CUDA == 1 and torch.cuda.is_available():
                    z = z.cuda()
                fake_img = netG(z)
                netD.zero_grad()
                real_out = netD(real_img).mean()
                fake_out = netD(fake_img).mean()

                d_loss = 1 - real_out + fake_out
                d_loss.backward(retain_graph=True)
                optimizerD.step()

                ############################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################
                # willy change:
                '''
                netG.zero_grad()
                g_loss = generator_criterion(fake_out, fake_img, real_img)

                optimizerG.step()

                fake_img = netG(z)
                fake_out = netD(fake_img).mean()

                g_loss.backward()
                '''

                netG.zero_grad()
                fake_img = netG(z)
                fake_out = netD(fake_img).mean()
                g_loss = generator_criterion(fake_out, fake_img, real_img)
                g_loss.backward()

                # loss for current batch before optimization
                running_results['g_loss'] += g_loss.item() * batch_size
                running_results['d_loss'] += d_loss.item() * batch_size
                running_results['d_score'] += real_out.item() * batch_size
                running_results['g_score'] += fake_out.item() * batch_size

                train_bar.set_description(
                    desc='epochs:[%d/%d] index:[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f batch_size%d '
                         % (epoch, NUM_EPOCHS, index, index_size,
                            running_results['d_loss'],
                            running_results['g_loss'],
                            running_results['d_score'],
                            running_results['g_score'],
                            running_results['batch_sizes']
                            )
                )

                '''
                train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    epoch, NUM_EPOCHS, 
                    running_results['d_loss'] / running_results['batch_sizes'],
                    running_results['g_loss'] / running_results['batch_sizes'],
                    running_results['d_score'] / running_results['batch_sizes'],
                    running_results['g_score'] / running_results['batch_sizes']))
                '''

        out_path = 'training_results/SRF_' + str(epoch) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # do eval
        netG.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            # for val_lr, val_hr_restore, val_hr in val_bar:
            for val_lrs, val_hrs, rgbs, path in val_bar:
                val_lr = val_lrs[index]
                val_hr = val_hrs[index]
                for index in range(len(datas)):
                    batch_size = val_lr.size(0)
                    valing_results['batch_sizes'] += batch_size
                    lr = val_lr
                    hr = val_hr
                    if USE_CUDA == 1 and torch.cuda.is_available():
                        lr = lr.cuda()
                        hr = hr.cuda()
                    sr = netG(lr)

                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    valing_results['ssims'] += batch_ssim * batch_size
                    valing_results['psnr'] = 10 * log10((hr.max() ** 2) / (valing_results['mse']
                                                                           / valing_results['batch_sizes']))
                    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                    val_bar.set_description(
                        desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                            valing_results['psnr'], valing_results['ssim']))

                    val_images.extend(
                        [display_transform()(
                            val_lr.squeeze(0)),
                            display_transform()(hr.data.cpu().squeeze(0)),
                            display_transform()(sr.data.cpu().squeeze(0))]
                    )
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        # torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netG.state_dict(), opt.epochs_path + 'netG_epoch_%d.pth' % epoch)
        torch.save(netD.state_dict(), opt.epochs_path + 'netD_epoch_%d.pth' % epoch)
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            # out_path = 'statistics/'
            out_path = opt.statistics_path
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(epoch) + '_train_results.csv', index_label='Epoch')
