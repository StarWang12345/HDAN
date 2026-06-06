import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from adamp import AdamP
from tensorboardX import SummaryWriter
from dataset import TrainDataset, TestDataset
import datetime as datetimes
import time as times
from util import *
import shutil

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()

time = datetimes.datetime.now().strftime('%m.%d-%H:%M:%S')
torch.manual_seed(42)
class Solver():
    def __init__(self, model, cfg):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        self.refiner = model(scale=cfg.scale)
        if cfg.loss_fn in ["MSE"]:
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]:
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()
        elif cfg.loss_fn in ["Charbonnier"]:
            self.loss_fn = CharbonnierLoss()

        self.optim = AdamP(
            filter(lambda p: p.requires_grad, self.refiner.parameters()),
            cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim,
            milestones=[100000, 200000, 350000, 450000],  # 只衰减两次[150000, 225000],30w-[60000,160000,240000],50w-milestones=[100000, 200000, 350000, 450000],  # 增加1次后期衰减
            gamma=0.5
        )
        self.train_data = TrainDataset(cfg.train_data_path,
                                       scale=cfg.scale,
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=0,##原本是0
                                       shuffle=False, drop_last=True)

        self.refiner = self.refiner.to(self.device)
        self.loss_fn = self.loss_fn
        self.folder_name = str(cfg.loss_fn) + '_' + str(cfg.batch_size) + '_' + str(cfg.max_steps)[0] + 'K' + '_' + \
                           str(cfg.lr) + '_' + str(cfg.scale)

        checkpoint_folder = 'logs/{}/checkpoints'.format(self.folder_name)
        mkdir(checkpoint_folder)

        if cfg.resume:
            PATH = os.path.join("logs", self.folder_name, "checkpoints")
            all_checkpoints = [os.path.join(PATH, f) for f in os.listdir(PATH) if f.endswith('.pth.tar')]
            all_checkpoints.sort(key=os.path.getmtime)

            if len(all_checkpoints) > 0:
                PATH = os.path.join(all_checkpoints[-1])
                print("=> loading checkpoint '{}'".format(PATH))
                checkpoint = torch.load(PATH)
                self.refiner.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                self.step = checkpoint['step']
                self.best_psnr = checkpoint["best_psnr"]
            else:
                print("=> no checkpoint at '{}'".format(PATH))
                self.best_psnr = 0
                self.step = 0
        else:
            self.best_psnr = 0
            self.step = 0

        self.cfg = cfg

        self.writer = SummaryWriter(log_dir=os.path.join("logs/{}/tensorboard/".format(self.folder_name)))
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("Number of parameters for scale X{}: {}".format(cfg.scale, num_params))

    def fit(self):
        cfg = self.cfg

        refiner = nn.DataParallel(self.refiner,
                                  device_ids=range(cfg.num_gpu))
        self.mean_content = 0.
        self.mean_l1 = 0.

        learning_rate = cfg.lr
        while True:
            for inputs in self.train_loader:

                self.refiner.train()
                total_loss = []

                scale = cfg.scale

                hr, lr = inputs[-1][0], inputs[-1][1]

                hr = hr.to(self.device)
                lr = lr.to(self.device)

                sr_main = refiner(lr, scale)#改动过原sr_main = refiner(lr, scale)

                loss = self.loss_fn(sr_main, hr)

                self.optim.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.refiner.parameters(), cfg.clip)
                self.optim.step()
                self.scheduler.step()

                self.mean_l1 += loss

                # learning_rate = self.decay_learning_rate()
                # for param_group in self.optim.param_groups:
                #     param_group["lr"] = learning_rate
                # self.scheduler.step()
                # learning_rate = self.optim.param_groups[0]['lr']

                self.step += 1
                sys.stdout.write("\r==>>Steps:[%d/ %d] Total:[%.6f] "
                                 % (self.step, cfg.max_steps, loss.item()))
                self.writer.add_scalar('Loss', loss.data.cpu().numpy(), global_step=self.step)

                if cfg.verbose and self.step % cfg.print_interval == 0:
                    with open('logs/{}/'.format(self.folder_name) + 'logs.txt', 'a') as f:
                        PATH = os.path.join('logs/{}/checkpoints/'.format(self.folder_name),
                                            "{}_{:06d}.pth.tar".format(cfg.ckpt_name, self.step))

                        t1 = times.time()

                        mean_psnr = self.evaluate(cfg.valid_data_path, scale=cfg.scale, num_step=self.step)
                        t2 = times.time()

                        self.writer.add_scalar("PSNR_{}x:".format(scale), mean_psnr, self.step)


                        print('-- PSNR_x{}: {:.5f}  -- Total_Loss: {:.5f}\n'
                                        .format(scale, mean_psnr, (self.mean_l1) / cfg.print_interval))

                        torch.save({'step': self.step, 'model_state_dict': self.refiner.state_dict(),
                                        'optimizer_state_dict': self.optim.state_dict(), 'best_psnr': self.best_psnr}, PATH)

                        f.write('Step: {}'
                                             '--> PSNR_x{}:{:.5f} -->{:.3f}m\n'
                                                .format(self.step, scale, mean_psnr, ((t2 - t1)/60)))
                        

                    self.mean_l1 = 0.
                    self.mean_content = 0.

                # if self.step % 400 == 0:  
                #     with torch.no_grad():
                #         _, _, mask = self.refiner(lr, scale)  

            if self.step > cfg.max_steps: break

    def evaluate(self, test_data_dir, scale, num_step=0, tile=64, overlap=8):
        self.refiner.eval()

        test_data = TestDataset(test_data_dir, scale=scale)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

        mean_psnr = 0
        with torch.no_grad():
            for _, (hr, lr, name) in enumerate(test_loader):
                lr, hr = lr.to(self.device), hr.to(self.device)
                N, C, h_lr, w_lr = lr.shape
                h_hr, w_hr = h_lr * scale, w_lr * scale

                # 输出缓冲区
                sr = torch.zeros((N, C, h_hr, w_hr), device=self.device)
                count = torch.zeros_like(sr)

            # 步长
                stride = tile - overlap
                h_idx_list = list(range(0, h_lr - tile + 1, stride)) or [0]
                w_idx_list = list(range(0, w_lr - tile + 1, stride)) or [0]
                if h_idx_list[-1] + tile < h_lr: h_idx_list.append(h_lr - tile)
                if w_idx_list[-1] + tile < w_lr: w_idx_list.append(w_lr - tile)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        lr_patch = lr[:, :, h_idx:h_idx + tile, w_idx:w_idx + tile]
                        with torch.cuda.amp.autocast():          # 可选：混合精度再省 30%
                            sr_patch = self.refiner(lr_patch, scale)
                        sr[:, :, h_idx * scale:(h_idx + tile) * scale,
                        w_idx * scale:(w_idx + tile) * scale] += sr_patch
                        count[:, :, h_idx * scale:(h_idx + tile) * scale,
                            w_idx * scale:(w_idx + tile) * scale] += 1

                sr /= count
                mean_psnr += calc_psnr(sr, hr, scale,1, benchmark=True) / len(test_data)

        return mean_psnr



    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_name):
        save_path = os.path.join(
            'logs/{}/checkpoints/'.format(self.folder_name), "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    # def decay_learning_rate(self):
    #     lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
    #     return lr

    def save_checkpoint(self, is_best, filename='checkpoint.pth.tar'):
        save_path = os.path.join(self.cfg.logdir, self.folder_name) + '/'
        torch.save(self.refiner, save_path + filename)
        if is_best:
            shutil.copyfile(save_path + filename, save_path + 'model_best.pth.tar')
