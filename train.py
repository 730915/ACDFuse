#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings('ignore')

from net import Decoder, Encoder, INNModule
from ImprovementModule.APCA import APCA
from utils.dataset import H5Dataset
from ImprovementModule.HierarchicalFusionLayer import HierarchicalFusionLayer
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia
from torch.utils.tensorboard import SummaryWriter
from ImprovementModule.APCA import APCA
import torch.fft as tfft
import kornia


def fft_mask(x, ratio=0.1, mode='low'):
    """频域掩码函数"""
    _, _, H, W = x.shape
    x_fft = torch.fft.fft2(x)
    x_fft_shift = torch.fft.fftshift(x_fft)

    mask = torch.zeros_like(x_fft_shift)
    center_h, center_w = H // 2, W // 2

    if mode == 'low':
        h_range = int(H * ratio)
        w_range = int(W * ratio)
        mask[:, :, center_h - h_range:center_h + h_range, center_w - w_range:center_w + w_range] = 1
    else:  # high
        mask = torch.ones_like(x_fft_shift)
        h_range = int(H * ratio)
        w_range = int(W * ratio)
        mask[:, :, center_h - h_range:center_h + h_range, center_w - w_range:center_w + w_range] = 0

    x_fft_masked = x_fft_shift * mask
    x_fft_ishift = torch.fft.ifftshift(x_fft_masked)
    x_filtered = torch.fft.ifft2(x_fft_ishift).real

    return x_filtered


def low_freq(x, ratio=0.1):
    """提取低频分量"""
    return fft_mask(x, ratio, 'low')


def high_freq(x, ratio=0.1):
    """提取高频分量"""
    return fft_mask(x, ratio, 'high')


def ortho_loss(detail, base, eps=1e-6):
    """正交损失函数"""
    detail_flat = detail.view(detail.size(0), -1)
    base_flat = base.view(base.size(0), -1)

    dot_product = torch.sum(detail_flat * base_flat, dim=1)
    detail_norm = torch.norm(detail_flat, dim=1) + eps
    base_norm = torch.norm(base_flat, dim=1) + eps

    cos_sim = dot_product / (detail_norm * base_norm)
    return torch.mean(cos_sim ** 2)


def edge_map(x):
    """边缘检测"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)

    edge_x = torch.nn.functional.conv2d(x, sobel_x, padding=1)
    edge_y = torch.nn.functional.conv2d(x, sobel_y, padding=1)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)

    return edge


def check_for_nan_inf(tensor, name="tensor"):
    """检查张量中的NaN和Inf值"""
    if torch.isnan(tensor).any():
        print(f"Warning: NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"Warning: Inf detected in {name}")
        return True
    return False


# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 初始化损失函数
criteria_fusion = Fusionloss()

model_str = 'ACDFuse'

# 训练参数 - 针对数值稳定性优化
num_epochs = 120
epoch_gap = 40

# 学习率设置 - 第二阶段使用更低的学习率
lr_phase1 = 1e-4  # 第一阶段学习率
lr_phase2 = 5e-5  # 第二阶段学习率（降低）

weight_decay = 1e-6  # 增加权重衰减
batch_size = 2
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']

# 损失系数
coeff_mse_loss_VF = 1.
coeff_mse_loss_IF = 1.
coeff_decomp = 2.
coeff_tv = 5.

# 梯度裁剪 - 更严格的梯度裁剪
clip_grad_norm_value = 0.005  # 降低梯度裁剪阈值

# 学习率调度参数
optim_step = 20
optim_gamma = 0.5

print('| model: {} | num_epochs: {} | batch_size: {} | lr_phase1: {} | lr_phase2: {} | GPU: {} |'.format(
    model_str, num_epochs, batch_size, lr_phase1, lr_phase2, GPU_number))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型初始化
DIDF_Encoder = nn.DataParallel(Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(APCA(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(INNModule(num_layers=1)).to(device)
HierarchicalFuser = nn.DataParallel(HierarchicalFusionLayer(dim=64, num_heads=4, num_layers=3, fusion_mode='sum')).to(
    device)

# 优化器初始化 - 第一阶段
optimizer1 = torch.optim.AdamW(DIDF_Encoder.parameters(), lr=lr_phase1, weight_decay=weight_decay, eps=1e-8)
optimizer2 = torch.optim.AdamW(DIDF_Decoder.parameters(), lr=lr_phase1, weight_decay=weight_decay, eps=1e-8)
optimizer3 = torch.optim.AdamW(BaseFuseLayer.parameters(), lr=lr_phase1, weight_decay=weight_decay, eps=1e-8)
optimizer4 = torch.optim.AdamW(DetailFuseLayer.parameters(), lr=lr_phase1, weight_decay=weight_decay, eps=1e-8)
optimizer5 = torch.optim.AdamW(HierarchicalFuser.parameters(), lr=lr_phase1, weight_decay=weight_decay, eps=1e-8)

# 学习率调度器
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)

# 损失函数
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')

# 数据加载器
trainloader = DataLoader(H5Dataset(r"data/MSRS_train/train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)
loader = {'train': trainloader}

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

# TensorBoard日志
writer = SummaryWriter(f'runs/stable_training_{timestamp}')

# 训练循环
step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()
epoch_losses = []

# 添加损失监控
loss_history = []
nan_count = 0

for epoch in range(num_epochs):
    epoch_loss = 0.0

    # 第二阶段切换学习率
    if epoch == epoch_gap:
        print(f"\n切换到第二阶段训练，降低学习率到 {lr_phase2}")
        for optimizer in [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_phase2

    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

        # 输入数据检查
        if check_for_nan_inf(data_VIS, "data_VIS") or check_for_nan_inf(data_IR, "data_IR"):
            print(f"Skipping batch {i} due to NaN/Inf in input data")
            continue

        # 设置训练模式
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()
        HierarchicalFuser.train()

        # 清零梯度
        for optimizer in [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]:
            optimizer.zero_grad()

        try:
            if epoch < epoch_gap:
                # 第一阶段训练
                feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
                feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)

                data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
                data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

                # 检查特征是否包含NaN
                if (check_for_nan_inf(feature_V_B, "feature_V_B") or
                        check_for_nan_inf(feature_V_D, "feature_V_D") or
                        check_for_nan_inf(feature_I_B, "feature_I_B") or
                        check_for_nan_inf(feature_I_D, "feature_I_D")):
                    print(f"Skipping batch {i} due to NaN/Inf in features")
                    continue

                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)

                mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
                mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)
                Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                       kornia.filters.SpatialGradient()(data_VIS_hat))
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

                lf_V = low_freq(feature_V_B)
                lf_I = low_freq(feature_I_B)
                L_LF = torch.mean(torch.abs(lf_V - lf_I))

                hf_V = high_freq(feature_V_D)
                lfB_V = low_freq(feature_V_B)
                hf_I = high_freq(feature_I_D)
                lfB_I = low_freq(feature_I_B)
                L_ortho = ortho_loss(hf_V, lfB_V) + ortho_loss(hf_I, lfB_I)

                alpha_decomp = coeff_decomp
                alpha_ortho = 0.5
                alpha_LF = 0.5

                loss = (coeff_mse_loss_VF * mse_loss_V +
                        coeff_mse_loss_IF * mse_loss_I +
                        alpha_decomp * loss_decomp +
                        coeff_tv * Gradient_loss +
                        alpha_ortho * L_ortho +
                        alpha_LF * L_LF)

                # 损失检查
                if check_for_nan_inf(loss, "loss_phase1"):
                    print(f"NaN/Inf detected in Phase 1 loss at epoch {epoch}, batch {i}")
                    nan_count += 1
                    continue

                loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

                optimizer1.step()
                optimizer2.step()

            else:
                # 第二阶段训练
                feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
                feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)

                # 检查特征
                if (check_for_nan_inf(feature_V_B, "feature_V_B_phase2") or
                        check_for_nan_inf(feature_V_D, "feature_V_D_phase2") or
                        check_for_nan_inf(feature_I_B, "feature_I_B_phase2") or
                        check_for_nan_inf(feature_I_D, "feature_I_D_phase2")):
                    print(f"Skipping batch {i} due to NaN/Inf in Phase 2 features")
                    continue

                feature_F_B = BaseFuseLayer(feature_I_B, feature_V_B)
                feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
                feature_F_Hierarchical = HierarchicalFuser(feature_F_B, feature_F_D)
                data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_Hierarchical, feature_F_Hierarchical)

                # 检查融合结果
                if check_for_nan_inf(data_Fuse, "data_Fuse"):
                    print(f"Skipping batch {i} due to NaN/Inf in fused image")
                    continue

                mse_loss_V = 5 * Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
                mse_loss_I = 5 * Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)

                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
                fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

                # 高频保真损失 - 添加数值稳定性
                HF_F = high_freq(data_Fuse)
                HF_V = high_freq(data_VIS)
                HF_I = high_freq(data_IR)
                L_HF = torch.mean(torch.abs(torch.abs(HF_F) - torch.max(torch.abs(HF_V), torch.abs(HF_I))))

                # IR显著保持损失
                W_ir = edge_map(data_IR)
                L_IRSal = torch.mean(torch.abs(W_ir * (data_Fuse - data_IR)))

                # 边缘一致性损失
                gradF = edge_map(data_Fuse)
                gradV = edge_map(data_VIS)
                gradI = edge_map(data_IR)
                L_edge = torch.mean(torch.abs(gradF - torch.max(gradV, gradI)))

                # 使用更保守的损失权重
                # beta_HF = 0.1    # 降低权重
                # beta_IR = 0.1    # 降低权重
                # beta_edge = 0.1  # 降低权重
                beta_HF = 0.3  # 降低权重
                beta_IR = 0.3  # 降低权重
                beta_edge = 0.3  # 降低权重

                loss = (fusionloss +
                        coeff_decomp * loss_decomp +
                        beta_HF * L_HF +
                        beta_IR * L_IRSal +
                        beta_edge * L_edge)

                # 损失检查
                if check_for_nan_inf(loss, "loss_phase2"):
                    print(f"NaN/Inf detected in Phase 2 loss at epoch {epoch}, batch {i}")
                    print(f"fusionloss: {fusionloss.item()}, loss_decomp: {loss_decomp.item()}")
                    print(f"L_HF: {L_HF.item()}, L_IRSal: {L_IRSal.item()}, L_edge: {L_edge.item()}")
                    nan_count += 1
                    continue

                loss.backward()

                # 更严格的梯度裁剪
                nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(HierarchicalFuser.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()
                optimizer5.step()

            # 记录损失
            epoch_loss += loss.item()
            loss_history.append(loss.item())

            # TensorBoard记录
            global_step = epoch * len(loader['train']) + i
            writer.add_scalar('Loss/Total', loss.item(), global_step)
            writer.add_scalar('Loss/Learning_Rate', optimizer1.param_groups[0]['lr'], global_step)

        except Exception as e:
            print(f"Error in epoch {epoch}, batch {i}: {str(e)}")
            nan_count += 1
            continue

        # 计算剩余时间
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # 打印训练进度
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [NaN count: %d] ETA: %.10s" % (
                epoch, num_epochs, i, len(loader['train']), loss.item(), nan_count, time_left,
            )
        )

    # 学习率调度
    scheduler1.step()
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()
        scheduler5.step()

    # 学习率下限
    for optimizer in [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]:
        if optimizer.param_groups[0]['lr'] <= 1e-7:
            optimizer.param_groups[0]['lr'] = 1e-7

    # 记录epoch损失
    avg_epoch_loss = epoch_loss / len(loader['train'])
    epoch_losses.append(avg_epoch_loss)
    writer.add_scalar('Loss/Epoch_Average', avg_epoch_loss, epoch)

    print(f"\nEpoch {epoch} completed. Average loss: {avg_epoch_loss:.6f}")

# 保存模型
if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
        'HierarchicalFuser': HierarchicalFuser.state_dict(),
        'epoch_losses': epoch_losses,
        'nan_count': nan_count,
        'loss_history': loss_history
    }
    model_path = f"models/ACDFuse_{timestamp}.pth"
    torch.save(checkpoint, model_path)
    print(f"稳定版本模型已保存到: {model_path}")
    print(f"训练过程中NaN出现次数: {nan_count}")

writer.close()