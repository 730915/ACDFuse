#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACDFuse + 检测头联合训练脚本 (CDFDet v2)

三阶段渐进训练策略:
- Stage I: 融合自编码器预训练 (继承原有)
- Stage II: 融合质量优化 (继承原有)
- Stage III-a: 联合预热 (冻结融合主干)
- Stage III-b: 全参数联合微调

核心设计:
- FADN: 融合特征 → YOLOv5 FPN 残差注入
- TAFFM: YOLOv5 FPN → APCA 注意力调制
- 不确定度自动加权联合损失
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

from net import (
    Encoder, Decoder, INNEncoder,
    APCA, APCA_TAFFM,
    HierarchicalFusionLayer,
    FADN, TAFFM, YOLOv5Head
)
from ImprovementModule.FADC import FADConv
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
# 检测损失 (简化版 YOLOv5 损失)
# ============================================================================

class YOLOv5Loss(nn.Module):
    """YOLOv5 简化检测损失"""
    def __init__(self, num_classes=6):
        super(YOLOv5Loss, self).__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, predictions, targets):
        """
        Args:
            predictions: 检测头输出列表 [[B, (num_classes+5)*3, H, W], ...]
            targets: 目标字典 {'boxes': [...], 'labels': [...]}
        Returns:
            总检测损失
        """
        # 简化实现: 返回预测值的分类和回归损失
        # 实际使用时建议使用 ultralytics YOLOv5 的完整损失
        total_loss = 0
        for pred in predictions:
            # 简化: 用预测值的范数作为辅助损失
            total_loss += torch.mean(pred ** 2) * 0.001
        return total_loss


# ============================================================================
# 不确定度加权联合损失 (Kendall CVPR 2018)
# ============================================================================

class UncertaintyWeightedLoss(nn.Module):
    """
    同方差不确定度自动加权联合损失
    理论基础: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018
    """
    def __init__(self, lambda_reg=0.01):
        super(UncertaintyWeightedLoss, self).__init__()
        self.lambda_reg = lambda_reg
        # 可学习的不确定度参数 (对数方差)
        self.s_fus = nn.Parameter(torch.tensor(0.0))  # 融合任务
        self.s_det = nn.Parameter(torch.tensor(0.0))  # 检测任务

    def forward(self, L_fus, L_det):
        """
        Args:
            L_fus: 融合损失
            L_det: 检测损失
        Returns:
            加权总损失
        """
        # 不确定度加权: 1/(2*e^s) * L + λ*s
        w_fus = 0.5 * torch.exp(-self.s_fus)
        w_det = 0.5 * torch.exp(-self.s_det)

        loss_fus_weighted = w_fus * L_fus + self.lambda_reg * self.s_fus
        loss_det_weighted = w_det * L_det + self.lambda_reg * self.s_det

        total_loss = loss_fus_weighted + loss_det_weighted

        return total_loss, w_fus, w_det


# ============================================================================
# 配置参数
# ============================================================================

# GPU设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 训练阶段
STAGE_I_EPOCHS = 40   # Stage I: 自编码器重建
STAGE_II_EPOCHS = 40  # Stage II: 融合质量优化
STAGE_IIIA_EPOCHS = 20  # Stage III-a: 联合预热
STAGE_IIIB_EPOCHS = 60  # Stage III-b: 全参数联合微调

TOTAL_EPOCHS = STAGE_I_EPOCHS + STAGE_II_EPOCHS + STAGE_IIIA_EPOCHS + STAGE_IIIB_EPOCHS

# 学习率设置
lr_stage1 = 1e-4   # Stage I, II
lr_stage3a = 1e-4  # Stage III-a
lr_stage3b_encoder = 1e-5  # Stage III-b 编码器 (低学习率保护)
lr_stage3b_other = 1e-4  # Stage III-b 其他模块
lr_uncertainty = 1e-3  # 不确定度参数
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

# 检测任务参数
NUM_CLASSES = 6  # M3FD: Bus/Car/Lamp/Motorcycle/People/Truck
FPN_CHANNELS = [128, 256, 512]  # YOLOv5 FPN 通道数

# 不确定度正则化系数
LAMBDA_REG = 0.01

print(f'| Total Epochs: {TOTAL_EPOCHS} | '
      f'Stage I: {STAGE_I_EPOCHS} | Stage II: {STAGE_II_EPOCHS} | '
      f'Stage III-a: {STAGE_IIIA_EPOCHS} | Stage III-b: {STAGE_IIIB_EPOCHS} |')
print(f'| batch_size: {batch_size} | GPU: {os.environ["CUDA_VISIBLE_DEVICES"]} |')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# 模型初始化
# ============================================================================

# ACDFuse 编码器 (双分支: SERTE + INN)
encoder = nn.DataParallel(Encoder()).to(device)

# ACDFuse 解码器 (包含 Bi-IRCSA)
decoder = nn.DataParallel(Decoder()).to(device)

# APCA (自适应位置感知交叉注意力模块)
apca_fuse = nn.DataParallel(APCA(dim=64, num_heads=8)).to(device)

# INN 融合层 (用于细节特征融合)
inn_fuse = nn.DataParallel(INNEncoder(num_layers=1)).to(device)

# HFL (分层融合层)
hfl_fuse = nn.DataParallel(HierarchicalFusionLayer(dim=64, num_heads=4, num_layers=3, fusion_mode='sum')).to(device)

# FADN (融合增强检测颈部模块)
fodn = nn.DataParallel(FADN(in_channels=64, fpn_channels=FPN_CHANNELS)).to(device)

# TAFFM (任务感知特征反馈模块)
taffm = nn.DataParallel(TAFFM(fpn_channels=FPN_CHANNELS)).to(device)

# YOLOv5 检测头 (简化版)
yolo_head = nn.DataParallel(YOLOv5Head(num_classes=NUM_CLASSES, fpn_channels=FPN_CHANNELS)).to(device)


# ============================================================================
# 优化器初始化
# ============================================================================

# ACDFuse 优化器
optimizer_encoder = torch.optim.AdamW(encoder.parameters(), lr=lr_stage1, weight_decay=weight_decay, eps=1e-8)
optimizer_decoder = torch.optim.AdamW(decoder.parameters(), lr=lr_stage1, weight_decay=weight_decay, eps=1e-8)
optimizer_apca = torch.optim.AdamW(apca_fuse.parameters(), lr=lr_stage1, weight_decay=weight_decay, eps=1e-8)
optimizer_inn = torch.optim.AdamW(inn_fuse.parameters(), lr=lr_stage1, weight_decay=weight_decay, eps=1e-8)
optimizer_hfl = torch.optim.AdamW(hfl_fuse.parameters(), lr=lr_stage1, weight_decay=weight_decay, eps=1e-8)

# 联合训练优化器 (Stage III)
optimizer_fodn = torch.optim.AdamW(fodn.parameters(), lr=lr_stage3b_other, weight_decay=weight_decay, eps=1e-8)
optimizer_taffm = torch.optim.AdamW(taffm.parameters(), lr=lr_stage3b_other, weight_decay=weight_decay, eps=1e-8)
optimizer_yolo = torch.optim.AdamW(yolo_head.parameters(), lr=lr_stage3b_other, weight_decay=weight_decay, eps=1e-8)

# 不确定度参数优化器
uncertainty_params = nn.ParameterDict({
    's_fus': nn.Parameter(torch.tensor(0.0)),
    's_det': nn.Parameter(torch.tensor(0.0))
})
optimizer_uncertainty = torch.optim.AdamW([uncertainty_params['s_fus'], uncertainty_params['s_det']],
                                           lr=lr_uncertainty, weight_decay=weight_decay, eps=1e-8)


# ============================================================================
# 学习率调度器
# ============================================================================

scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=20, gamma=0.5)
scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer_decoder, step_size=20, gamma=0.5)
scheduler_apca = torch.optim.lr_scheduler.StepLR(optimizer_apca, step_size=20, gamma=0.5)
scheduler_inn = torch.optim.lr_scheduler.StepLR(optimizer_inn, step_size=20, gamma=0.5)
scheduler_hfl = torch.optim.lr_scheduler.StepLR(optimizer_hfl, step_size=20, gamma=0.5)


# ============================================================================
# 损失函数
# ============================================================================

criteria_fusion = Fusionloss()
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')
yolo_loss_fn = YOLOv5Loss(num_classes=NUM_CLASSES)
uncertainty_loss_fn = UncertaintyWeightedLoss(lambda_reg=LAMBDA_REG)


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

timestamp = datetime.datetime.now().strftime("%m-%d-%d-%H-%M")
writer = SummaryWriter(f'runs/joint_training_{timestamp}')


# ============================================================================
# 辅助函数: 创建虚拟 FPN 特征 (Stage III 预热用)
# ============================================================================

def create_dummy_fpn(batch_size, fpn_channels=[128, 256, 512], device='cuda'):
    """创建虚拟 FPN 特征用于 Stage III-a 预热"""
    h, w = 80, 80  # P3 分辨率 (输入 640x640)
    p3 = torch.randn(batch_size, fpn_channels[0], h, w, device=device)
    p4 = torch.randn(batch_size, fpn_channels[1], h//2, w//2, device=device)
    p5 = torch.randn(batch_size, fpn_channels[2], h//4, w//4, device=device)
    return [p3, p4, p5]


# ============================================================================
# 训练循环
# ============================================================================

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()
epoch_losses = []
nan_count = 0

# 当前阶段
current_stage = 'I'


def set_stage(stage_name):
    """设置训练阶段和对应的学习率"""
    global current_stage
    current_stage = stage_name
    print(f"\n{'='*60}")
    print(f"切换到 {stage_name} 阶段")
    print(f"{'='*60}")


for epoch in range(TOTAL_EPOCHS):
    epoch_loss = 0.0
    epoch_loss_fus = 0.0
    epoch_loss_det = 0.0

    # 阶段切换
    if epoch == STAGE_I_EPOCHS:
        set_stage('II')
    elif epoch == STAGE_I_EPOCHS + STAGE_II_EPOCHS:
        set_stage('III-a')
        # Stage III-a: 冻结 ACDFuse 主干
        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        for param in apca_fuse.parameters():
            param.requires_grad = False
        for param in inn_fuse.parameters():
            param.requires_grad = False
        for param in hfl_fuse.parameters():
            param.requires_grad = False
        # 更新学习率
        for param_group in optimizer_fodn.param_groups:
            param_group['lr'] = lr_stage3a
        for param_group in optimizer_taffm.param_groups:
            param_group['lr'] = lr_stage3a
        for param_group in optimizer_yolo.param_groups:
            param_group['lr'] = lr_stage3a
    elif epoch == STAGE_I_EPOCHS + STAGE_II_EPOCHS + STAGE_IIIA_EPOCHS:
        set_stage('III-b')
        # Stage III-b: 解冻 ACDFuse 主干
        for param in encoder.parameters():
            param.requires_grad = True
        for param in decoder.parameters():
            param.requires_grad = True
        for param in apca_fuse.parameters():
            param.requires_grad = True
        for param in inn_fuse.parameters():
            param.requires_grad = True
        for param in hfl_fuse.parameters():
            param.requires_grad = True
        # 降低编码器学习率，保护预训练权重
        for param_group in optimizer_encoder.param_groups:
            param_group['lr'] = lr_stage3b_encoder
        for param_group in optimizer_decoder.param_groups:
            param_group['lr'] = lr_stage3b_encoder
        # 其他模块保持较高学习率
        for param_group in optimizer_apca.param_groups:
            param_group['lr'] = lr_stage3b_other
        for param_group in optimizer_inn.param_groups:
            param_group['lr'] = lr_stage3b_other
        for param_group in optimizer_hfl.param_groups:
            param_group['lr'] = lr_stage3b_other

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
        fodn.train()
        taffm.train()
        yolo_head.train()

        # 清零梯度
        for opt in [optimizer_encoder, optimizer_decoder, optimizer_apca,
                    optimizer_inn, optimizer_hfl, optimizer_fodn,
                    optimizer_taffm, optimizer_yolo, optimizer_uncertainty]:
            opt.zero_grad()

        try:
            if current_stage in ['I', 'II']:
                # =====================================================================
                # Stage I & II: 纯融合训练 (继承原有逻辑)
                # =====================================================================

                if current_stage == 'I':
                    # Stage I: 编码器-解码器重构
                    base_V, detail_V, shallow_V = encoder(data_VIS)
                    base_I, detail_I, shallow_I = encoder(data_IR)

                    recon_V, _ = decoder(data_VIS, base_V, detail_V)
                    recon_I, _ = decoder(data_IR, base_I, detail_I)

                    mse_loss_V = 5 * Loss_ssim(data_VIS, recon_V) + MSELoss(data_VIS, recon_V)
                    mse_loss_I = 5 * Loss_ssim(data_IR, recon_I) + MSELoss(data_IR, recon_I)

                    Gradient_loss = L1Loss(
                        kornia.filters.SpatialGradient()(data_VIS),
                        kornia.filters.SpatialGradient()(recon_V)
                    )

                    cc_loss_B = cc(base_V, base_I)
                    cc_loss_D = cc(detail_V, detail_I)
                    loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

                    lf_V = low_freq(base_V)
                    lf_I = low_freq(base_I)
                    L_LF = torch.mean(torch.abs(lf_V - lf_I))

                    hf_V = high_freq(detail_V)
                    lfB_V = low_freq(base_V)
                    hf_I = high_freq(detail_I)
                    lfB_I = low_freq(base_I)
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

                    epoch_loss_fus += loss.item()

                else:
                    # Stage II: 完整融合网络
                    base_V, detail_V, shallow_V = encoder(data_VIS)
                    base_I, detail_I, shallow_I = encoder(data_IR)

                    base_F = apca_fuse(base_I, base_V)
                    detail_F = inn_fuse(detail_V + detail_I)
                    fused_F = hfl_fuse(base_F, detail_F)

                    fused_img, _ = decoder(data_VIS, fused_F, fused_F)

                    mse_loss_V = 5 * Loss_ssim(data_VIS, fused_img) + MSELoss(data_VIS, fused_img)
                    mse_loss_I = 5 * Loss_ssim(data_IR, fused_img) + MSELoss(data_IR, fused_img)

                    cc_loss_B = cc(base_V, base_I)
                    cc_loss_D = cc(detail_V, detail_I)
                    loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

                    fusion_loss, _, _ = criteria_fusion(data_VIS, data_IR, fused_img)

                    HF_F = high_freq(fused_img)
                    HF_V = high_freq(data_VIS)
                    HF_I = high_freq(data_IR)
                    L_HF = torch.mean(torch.abs(torch.abs(HF_F) - torch.max(torch.abs(HF_V), torch.abs(HF_I))))

                    W_ir = edge_map(data_IR)
                    L_IRSal = torch.mean(torch.abs(W_ir * (fused_img - data_IR)))

                    gradF = edge_map(fused_img)
                    gradV = edge_map(data_VIS)
                    gradI = edge_map(data_IR)
                    L_edge = torch.mean(torch.abs(gradF - torch.max(gradV, gradI)))

                    beta_HF = 0.3
                    beta_IR = 0.3
                    beta_edge = 0.3

                    loss = (fusion_loss +
                            coeff_decomp * loss_decomp +
                            beta_HF * L_HF +
                            beta_IR * L_IRSal +
                            beta_edge * L_edge)

                    epoch_loss_fus += loss.item()

                # 损失检查
                if check_for_nan_inf(loss, f"loss_{current_stage}"):
                    nan_count += 1
                    continue

                loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

                optimizer_encoder.step()
                optimizer_decoder.step()
                if current_stage == 'II':
                    optimizer_apca.step()
                    optimizer_inn.step()
                    optimizer_hfl.step()

            else:
                # =====================================================================
                # Stage III: 联合训练 (融合 + 检测)
                # =====================================================================

                # 提取双分支特征
                base_V, detail_V, shallow_V = encoder(data_VIS)
                base_I, detail_I, shallow_I = encoder(data_IR)

                # 跨模态融合
                base_F = apca_fuse(base_I, base_V)
                detail_F = inn_fuse(detail_V + detail_I)
                fused_F = hfl_fuse(base_F, detail_F)

                # 生成融合图像
                fused_img, _ = decoder(data_VIS, fused_F, fused_F)

                # Stage III: 创建虚拟 FPN 特征 (实际应用中应使用真实 YOLOv5 FPN)
                # TODO: 集成真实 YOLOv5 FPN
                dummy_fpn = create_dummy_fpn(batch_size, FPN_CHANNELS, device)

                # FADN: 融合特征 → FPN 增强
                enhanced_fpn = fodn(fused_F, fused_F, dummy_fpn)

                # TAFFM: FPN → 注意力调制
                m_task = taffm(enhanced_fpn, target_size=(fused_F.shape[2], fused_F.shape[3]))

                # 检测头前向
                det_outputs = yolo_head(enhanced_fpn)

                # =====================================================================
                # 融合损失计算
                # =====================================================================
                mse_loss_V = 5 * Loss_ssim(data_VIS, fused_img) + MSELoss(data_VIS, fused_img)
                mse_loss_I = 5 * Loss_ssim(data_IR, fused_img) + MSELoss(data_IR, fused_img)

                cc_loss_B = cc(base_V, base_I)
                cc_loss_D = cc(detail_V, detail_I)
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

                fusion_loss, _, _ = criteria_fusion(data_VIS, data_IR, fused_img)

                HF_F = high_freq(fused_img)
                HF_V = high_freq(data_VIS)
                HF_I = high_freq(data_IR)
                L_HF = torch.mean(torch.abs(torch.abs(HF_F) - torch.max(torch.abs(HF_V), torch.abs(HF_I))))

                W_ir = edge_map(data_IR)
                L_IRSal = torch.mean(torch.abs(W_ir * (fused_img - data_IR)))

                gradF = edge_map(fused_img)
                gradV = edge_map(data_VIS)
                gradI = edge_map(data_IR)
                L_edge = torch.mean(torch.abs(gradF - torch.max(gradV, gradI)))

                # L_guide: 目标感知融合引导损失 (用检测置信度加权)
                # 简化: 用融合图像与源图像的差异作为代理
                W_guide = torch.sigmoid(torch.mean(fused_img, dim=[2, 3]))  # 简化版
                L_guide = torch.mean(W_guide * torch.abs(fused_img - data_IR))

                beta_HF = 0.3
                beta_IR = 0.3
                beta_edge = 0.3
                beta_guide = 0.1

                L_fus = (fusion_loss +
                         coeff_decomp * loss_decomp +
                         beta_HF * L_HF +
                         beta_IR * L_IRSal +
                         beta_edge * L_edge +
                         beta_guide * L_guide)

                epoch_loss_fus += L_fus.item()

                # =====================================================================
                # 检测损失计算
                # =====================================================================
                # TODO: 集成真实检测标注计算检测损失
                L_det = yolo_loss_fn(det_outputs, None)

                epoch_loss_det += L_det.item()

                # =====================================================================
                # 不确定度加权联合损失
                # =====================================================================
                s_fus = uncertainty_params['s_fus']
                s_det = uncertainty_params['s_det']

                w_fus = 0.5 * torch.exp(-s_fus)
                w_det = 0.5 * torch.exp(-s_det)

                loss_fus_weighted = w_fus * L_fus + LAMBDA_REG * s_fus
                loss_det_weighted = w_det * L_det + LAMBDA_REG * s_det
                loss = loss_fus_weighted + loss_det_weighted

                # 损失检查
                if check_for_nan_inf(loss, f"loss_{current_stage}"):
                    nan_count += 1
                    continue

                loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(apca_fuse.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(inn_fuse.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(hfl_fuse.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(fodn.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(taffm.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(yolo_head.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

                optimizer_encoder.step()
                optimizer_decoder.step()
                optimizer_apca.step()
                optimizer_inn.step()
                optimizer_hfl.step()
                optimizer_fodn.step()
                optimizer_taffm.step()
                optimizer_yolo.step()
                optimizer_uncertainty.step()

            # 记录损失
            epoch_loss += loss.item()

            # TensorBoard记录
            global_step = epoch * len(loader['train']) + i
            writer.add_scalar('Loss/Total', loss.item(), global_step)
            writer.add_scalar('Loss/Learning_Rate', optimizer_encoder.param_groups[0]['lr'], global_step)
            writer.add_scalar(f'Loss/{current_stage}_Fus', epoch_loss_fus / max(i, 1), global_step)
            if current_stage.startswith('III'):
                writer.add_scalar(f'Loss/{current_stage}_Det', epoch_loss_det / max(i, 1), global_step)
                writer.add_scalar('Uncertainty/s_fus', s_fus.item() if 's_fus' in dir() else 0, global_step)
                writer.add_scalar('Uncertainty/s_det', s_det.item() if 's_det' in dir() else 0, global_step)

        except Exception as e:
            print(f"Error in epoch {epoch}, batch {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            nan_count += 1
            continue

        # 计算剩余时间
        batches_done = epoch * len(loader['train']) + i
        batches_left = TOTAL_EPOCHS * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # 打印训练进度
        sys.stdout.write(
            f"\r[Epoch {epoch}/{TOTAL_EPOCHS}] [Stage {current_stage}] "
            f"[Batch {i}/{len(loader['train'])}] "
            f"[loss: {loss.item():.6f}] [NaN: {nan_count}] ETA: {time_left}"
        )

    # 学习率调度
    scheduler_encoder.step()
    scheduler_decoder.step()
    if current_stage == 'II':
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

    print(f"\nEpoch {epoch} ({current_stage}) completed. Avg loss: {avg_epoch_loss:.6f}")


# ============================================================================
# 保存模型
# ============================================================================

checkpoint = {
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'apca_fuse': apca_fuse.state_dict(),
    'inn_fuse': inn_fuse.state_dict(),
    'hfl_fuse': hfl_fuse.state_dict(),
    'fodn': fodn.state_dict(),
    'taffm': taffm.state_dict(),
    'yolo_head': yolo_head.state_dict(),
    'uncertainty_params': {
        's_fus': uncertainty_params['s_fus'].item(),
        's_det': uncertainty_params['s_det'].item()
    },
    'epoch_losses': epoch_losses,
    'nan_count': nan_count,
}
model_path = f"models/ACDFuse_CDFDet_{timestamp}.pth"
torch.save(checkpoint, model_path)
print(f"模型已保存到: {model_path}")
print(f"训练过程中NaN出现次数: {nan_count}")

writer.close()
