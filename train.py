#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACDFuse 训练脚本

两阶段训练策略:
- 第一阶段: 优化自编码器结构中的编码器和解码器（图像重建）
- 第二阶段: 整个融合过程，生成高质量融合图像
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import time
import datetime
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import kornia

from net import Encoder, Decoder, INNEncoder
from ImprovementModule.APCA import APCA
from ImprovementModule.HierarchicalFusionLayer import HierarchicalFusionLayer
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, cc


# ============================================================================
# 频域处理函数
# ============================================================================

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
    else:
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


# ============================================================================
# 损失函数
# ============================================================================

def ortho_loss(detail, base, eps=1e-6):
    """正交损失函数 - 频率域正交解耦约束"""
    detail_flat = detail.view(detail.size(0), -1)
    base_flat = base.view(base.size(0), -1)

    dot_product = torch.sum(detail_flat * base_flat, dim=1)
    detail_norm = torch.norm(detail_flat, dim=1) + eps
    base_norm = torch.norm(base_flat, dim=1) + eps

    cos_sim = dot_product / (detail_norm * base_norm)
    return torch.mean(cos_sim ** 2)


def edge_map(x):
    """边缘检测 (Sobel算子)"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)

    edge_x = F.conv2d(x, sobel_x, padding=1)
    edge_y = F.conv2d(x, sobel_y, padding=1)
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


# ============================================================================
# 配置参数
# ============================================================================

# GPU设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 训练参数
model_name = 'ACDFuse'
num_epochs = 120
epoch_gap = 40  # 第一阶段到第二阶段的切换点

# 学习率设置
lr_phase1 = 1e-4  # 第一阶段学习率
lr_phase2 = 5e-5  # 第二阶段学习率
weight_decay = 1e-6

# 批大小
batch_size = 2

# 损失系数
coeff_mse_loss_VF = 1.0
coeff_mse_loss_IF = 1.0
coeff_decomp = 2.0
coeff_tv = 5.0

# 梯度裁剪
clip_grad_norm_value = 0.005

# 学习率调度
optim_step = 20
optim_gamma = 0.5

print(f'| model: {model_name} | num_epochs: {num_epochs} | batch_size: {batch_size} | '
      f'lr_phase1: {lr_phase1} | lr_phase2: {lr_phase2} | GPU: {os.environ["CUDA_VISIBLE_DEVICES"]} |')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# 模型初始化
# ============================================================================

# 编码器 (双分支: SERTE + INN)
encoder = nn.DataParallel(Encoder()).to(device)

# 解码器 (包含 Bi-IRCSA)
decoder = nn.DataParallel(Decoder()).to(device)

# APCA (自适应位置感知交叉注意力模块)
apca_fuse = nn.DataParallel(APCA(dim=64, num_heads=8)).to(device)

# INN 融合层 (用于细节特征融合)
inn_fuse = nn.DataParallel(INNEncoder(num_layers=1)).to(device)

# HFL (分层融合层)
hfl_fuse = nn.DataParallel(HierarchicalFusionLayer(dim=64, num_heads=4, num_layers=3, fusion_mode='sum')).to(device)


# ============================================================================
# 优化器初始化
# ============================================================================

optimizer_encoder = torch.optim.AdamW(encoder.parameters(), lr=lr_phase1, weight_decay=weight_decay, eps=1e-8)
optimizer_decoder = torch.optim.AdamW(decoder.parameters(), lr=lr_phase1, weight_decay=weight_decay, eps=1e-8)
optimizer_apca = torch.optim.AdamW(apca_fuse.parameters(), lr=lr_phase1, weight_decay=weight_decay, eps=1e-8)
optimizer_inn = torch.optim.AdamW(inn_fuse.parameters(), lr=lr_phase1, weight_decay=weight_decay, eps=1e-8)
optimizer_hfl = torch.optim.AdamW(hfl_fuse.parameters(), lr=lr_phase1, weight_decay=weight_decay, eps=1e-8)

# 学习率调度器
scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=optim_step, gamma=optim_gamma)
scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer_decoder, step_size=optim_step, gamma=optim_gamma)
scheduler_apca = torch.optim.lr_scheduler.StepLR(optimizer_apca, step_size=optim_step, gamma=optim_gamma)
scheduler_inn = torch.optim.lr_scheduler.StepLR(optimizer_inn, step_size=optim_step, gamma=optim_gamma)
scheduler_hfl = torch.optim.lr_scheduler.StepLR(optimizer_hfl, step_size=optim_step, gamma=optim_gamma)


# ============================================================================
# 损失函数
# ============================================================================

criteria_fusion = Fusionloss()
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


# ============================================================================
# 数据加载
# ============================================================================

trainloader = DataLoader(
    H5Dataset(r"data/MSRS_train/train_imgsize_128_stride_200.h5"),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
loader = {'train': trainloader}


# ============================================================================
# TensorBoard 日志
# ============================================================================

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
writer = SummaryWriter(f'runs/stable_training_{timestamp}')


# ============================================================================
# 训练循环
# ============================================================================

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()
epoch_losses = []
loss_history = []
nan_count = 0

for epoch in range(num_epochs):
    epoch_loss = 0.0

    # 第二阶段切换学习率
    if epoch == epoch_gap:
        print(f"\n切换到第二阶段训练，降低学习率到 {lr_phase2}")
        for opt in [optimizer_encoder, optimizer_decoder, optimizer_apca, optimizer_inn, optimizer_hfl]:
            for param_group in opt.param_groups:
                param_group['lr'] = lr_phase2

    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

        # 输入数据检查
        if check_for_nan_inf(data_VIS, "data_VIS") or check_for_nan_inf(data_IR, "data_IR"):
            print(f"Skipping batch {i} due to NaN/Inf in input data")
            continue

        # 设置训练模式
        encoder.train()
        decoder.train()
        apca_fuse.train()
        inn_fuse.train()
        hfl_fuse.train()

        # 清零梯度
        for opt in [optimizer_encoder, optimizer_decoder, optimizer_apca, optimizer_inn, optimizer_hfl]:
            opt.zero_grad()

        try:
            if epoch < epoch_gap:
                # =====================================================================
                # 第一阶段训练: 编码器-解码器重构
                # =====================================================================

                # 提取可见光和红外图像的特征
                base_V, detail_V, shallow_V = encoder(data_VIS)
                base_I, detail_I, shallow_I = encoder(data_IR)

                # 重构图像
                recon_V, _ = decoder(data_VIS, base_V, detail_V)
                recon_I, _ = decoder(data_IR, base_I, detail_I)

                # 检查特征
                if (check_for_nan_inf(base_V, "base_V") or check_for_nan_inf(detail_V, "detail_V") or
                    check_for_nan_inf(base_I, "base_I") or check_for_nan_inf(detail_I, "detail_I")):
                    print(f"Skipping batch {i} due to NaN/Inf in features")
                    continue

                # 重建损失
                mse_loss_V = 5 * Loss_ssim(data_VIS, recon_V) + MSELoss(data_VIS, recon_V)
                mse_loss_I = 5 * Loss_ssim(data_IR, recon_I) + MSELoss(data_IR, recon_I)

                # 梯度损失
                Gradient_loss = L1Loss(
                    kornia.filters.SpatialGradient()(data_VIS),
                    kornia.filters.SpatialGradient()(recon_V)
                )

                # 分解损失 (模态一致性)
                cc_loss_B = cc(base_V, base_I)
                cc_loss_D = cc(detail_V, detail_I)
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

                # 低频一致性损失
                lf_V = low_freq(base_V)
                lf_I = low_freq(base_I)
                L_LF = torch.mean(torch.abs(lf_V - lf_I))

                # 正交损失 (频率域解耦)
                hf_V = high_freq(detail_V)
                lfB_V = low_freq(base_V)
                hf_I = high_freq(detail_I)
                lfB_I = low_freq(base_I)
                L_ortho = ortho_loss(hf_V, lfB_V) + ortho_loss(hf_I, lfB_I)

                # 总损失
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
                nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

                optimizer_encoder.step()
                optimizer_decoder.step()

            else:
                # =====================================================================
                # 第二阶段训练: 完整融合网络
                # =====================================================================

                # 提取双分支特征
                base_V, detail_V, shallow_V = encoder(data_VIS)
                base_I, detail_I, shallow_I = encoder(data_IR)

                # 检查特征
                if (check_for_nan_inf(base_V, "base_V_phase2") or check_for_nan_inf(detail_V, "detail_V_phase2") or
                    check_for_nan_inf(base_I, "base_I_phase2") or check_for_nan_inf(detail_I, "detail_I_phase2")):
                    print(f"Skipping batch {i} due to NaN/Inf in Phase 2 features")
                    continue

                # 跨模态融合
                # 1. APCA 融合基础特征 (base_V, base_I -> base_F)
                base_F = apca_fuse(base_I, base_V)

                # 2. INN 融合细节特征 (detail_V + detail_I -> detail_F)
                detail_F = inn_fuse(detail_V + detail_I)

                # 3. HFL 分层融合
                fused_F = hfl_fuse(base_F, detail_F)

                # 4. 解码器生成融合图像
                fused_img, _ = decoder(data_VIS, fused_F, fused_F)

                # 检查融合结果
                if check_for_nan_inf(fused_img, "fused_img"):
                    print(f"Skipping batch {i} due to NaN/Inf in fused image")
                    continue

                # 融合损失
                mse_loss_V = 5 * Loss_ssim(data_VIS, fused_img) + MSELoss(data_VIS, fused_img)
                mse_loss_I = 5 * Loss_ssim(data_IR, fused_img) + MSELoss(data_IR, fused_img)

                # 分解损失
                cc_loss_B = cc(base_V, base_I)
                cc_loss_D = cc(detail_V, detail_I)
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

                # 融合损失函数
                fusion_loss, _, _ = criteria_fusion(data_VIS, data_IR, fused_img)

                # 高频保真损失
                HF_F = high_freq(fused_img)
                HF_V = high_freq(data_VIS)
                HF_I = high_freq(data_IR)
                L_HF = torch.mean(torch.abs(torch.abs(HF_F) - torch.max(torch.abs(HF_V), torch.abs(HF_I))))

                # IR显著保持损失
                W_ir = edge_map(data_IR)
                L_IRSal = torch.mean(torch.abs(W_ir * (fused_img - data_IR)))

                # 边缘一致性损失
                gradF = edge_map(fused_img)
                gradV = edge_map(data_VIS)
                gradI = edge_map(data_IR)
                L_edge = torch.mean(torch.abs(gradF - torch.max(gradV, gradI)))

                # 损失权重
                beta_HF = 0.3
                beta_IR = 0.3
                beta_edge = 0.3

                # 总损失
                loss = (fusion_loss +
                        coeff_decomp * loss_decomp +
                        beta_HF * L_HF +
                        beta_IR * L_IRSal +
                        beta_edge * L_edge)

                # 损失检查
                if check_for_nan_inf(loss, "loss_phase2"):
                    print(f"NaN/Inf detected in Phase 2 loss at epoch {epoch}, batch {i}")
                    print(f"fusion_loss: {fusion_loss.item()}, loss_decomp: {loss_decomp.item()}")
                    print(f"L_HF: {L_HF.item()}, L_IRSal: {L_IRSal.item()}, L_edge: {L_edge.item()}")
                    nan_count += 1
                    continue

                loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(apca_fuse.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(inn_fuse.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(hfl_fuse.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

                optimizer_encoder.step()
                optimizer_decoder.step()
                optimizer_apca.step()
                optimizer_inn.step()
                optimizer_hfl.step()

            # 记录损失
            epoch_loss += loss.item()
            loss_history.append(loss.item())

            # TensorBoard记录
            global_step = epoch * len(loader['train']) + i
            writer.add_scalar('Loss/Total', loss.item(), global_step)
            writer.add_scalar('Loss/Learning_Rate', optimizer_encoder.param_groups[0]['lr'], global_step)

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
            f"\r[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(loader['train'])}] "
            f"[loss: {loss.item():.6f}] [NaN count: {nan_count}] ETA: {time_left}"
        )

    # 学习率调度
    scheduler_encoder.step()
    scheduler_decoder.step()
    if not epoch < epoch_gap:
        scheduler_apca.step()
        scheduler_inn.step()
        scheduler_hfl.step()

    # 学习率下限
    for opt in [optimizer_encoder, optimizer_decoder, optimizer_apca, optimizer_inn, optimizer_hfl]:
        if opt.param_groups[0]['lr'] <= 1e-7:
            opt.param_groups[0]['lr'] = 1e-7

    # 记录epoch损失
    avg_epoch_loss = epoch_loss / len(loader['train'])
    epoch_losses.append(avg_epoch_loss)
    writer.add_scalar('Loss/Epoch_Average', avg_epoch_loss, epoch)

    print(f"\nEpoch {epoch} completed. Average loss: {avg_epoch_loss:.6f}")


# ============================================================================
# 保存模型
# ============================================================================

if True:
    checkpoint = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'apca_fuse': apca_fuse.state_dict(),
        'inn_fuse': inn_fuse.state_dict(),
        'hfl_fuse': hfl_fuse.state_dict(),
        'epoch_losses': epoch_losses,
        'nan_count': nan_count,
        'loss_history': loss_history
    }
    model_path = f"models/ACDFuse_{timestamp}.pth"
    torch.save(checkpoint, model_path)
    print(f"模型已保存到: {model_path}")
    print(f"训练过程中NaN出现次数: {nan_count}")

writer.close()
