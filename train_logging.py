#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import math
import numpy as np
from model import *
import config as c
from tensorboardX import SummaryWriter
import datasets
import viz
import modules.Unet_common as common
import warnings
import logging
import util
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


#####################
# Model initialize: #
#####################
net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
para = get_parameter_number(net)
print(para)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

dwt = common.DWT()
iwt = common.IWT()

if c.tain_next:
    load(c.MODEL_PATH + c.suffix)

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

util.setup_logger('train', '/home/jjp/HiNet-main2/', 'train_', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train')
logger_train.info(net)

try:
    writer = SummaryWriter(comment='hinet', filename_suffix="steg")

    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []
        g_loss_history = []
        r_loss_history = []
        l_loss_history = []
        #################
        #     train:    #
        #################

        for i_batch, data in enumerate(datasets.trainloader):
            data = data.to(device)
            cover = data[data.shape[0] // 2:]
            secret = data[:data.shape[0] // 2]
            cover_input = dwt(cover)
            secret_input = dwt(secret)

            input_img = torch.cat((cover_input, secret_input), 1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg_img = iwt(output_steg)

            #################
            #   backward:   #
            #################

            output_z_guass = gauss_noise(output_z.shape)

            output_rev = torch.cat((output_steg, output_z_guass), 1)
            output_image = net(output_rev, rev=True)

            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev)

            #################
            #     loss:     #
            #################
            g_loss = guide_loss(steg_img.cuda(), cover.cuda())
            r_loss = reconstruction_loss(secret_rev, secret)
            steg_low = output_steg.narrow(1, 0, c.channels_in)
            cover_low = cover_input.narrow(1, 0, c.channels_in)
            l_loss = low_frequency_loss(steg_low, cover_low)

            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])

            g_loss_history.append([g_loss.item(), 0.])
            r_loss_history.append([r_loss.item(), 0.])
            l_loss_history.append([l_loss.item(), 0.])

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        r_epoch_losses = np.mean(np.array(r_loss_history), axis=0)
        g_epoch_losses = np.mean(np.array(g_loss_history), axis=0)
        l_epoch_losses = np.mean(np.array(l_loss_history), axis=0)

        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])
        #################
        #     val:    #
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                psnr_s = []
                psnr_c = []
                net.eval()
                for x in datasets.testloader:
                    x = x.to(device)
                    cover = x[x.shape[0] // 2:, :, :, :]
                    secret = x[:x.shape[0] // 2, :, :, :]
                    cover_input = dwt(cover)
                    secret_input = dwt(secret)

                    input_img = torch.cat((cover_input, secret_input), 1)

                    #################
                    #    forward:   #
                    #################
                    output = net(input_img)
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)
                    steg = iwt(output_steg)
                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                    output_z = gauss_noise(output_z.shape)

                    #################
                    #   backward:   #
                    #################
                    output_steg = output_steg.cuda()
                    output_rev = torch.cat((output_steg, output_z), 1)
                    output_image = net(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                    secret_rev = iwt(secret_rev)

                    secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    steg = steg.cpu().numpy().squeeze() * 255
                    np.clip(steg, 0, 255)
                    psnr_temp = computePSNR(secret_rev, secret)
                    psnr_s.append(psnr_temp)
                    psnr_temp_c = computePSNR(cover, steg)
                    psnr_c.append(psnr_temp_c)

                writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, i_epoch)
                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)
                logger_train.info(
                    f"TEST:   "
                    f'PSNR_S: {np.mean(psnr_s):.4f} | '
                    f'PSNR_C: {np.mean(psnr_c):.4f} | '
                )
        viz.show_loss(epoch_losses)
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

        logger_train.info(f"Learning rate: {optim.param_groups[0]['lr']}")
        logger_train.info(
            f"Train epoch {i_epoch}:   "
            f'Loss: {epoch_losses[0].item():.4f} | '
            f'r_Loss: {r_epoch_losses[0].item():.4f} | '
            f'g_Loss: {g_epoch_losses[0].item():.4f} | '
            f'l_Loss: {l_epoch_losses[0].item():.4f} | '
        )

        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt')
        weight_scheduler.step()

    torch.save({'opt': optim.state_dict(),
                'net': net.state_dict()}, c.MODEL_PATH + 'model' + '.pt')
    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, c.MODEL_PATH + 'model_ABORT' + '.pt')
    raise

finally:
    viz.signal_stop()
