"""
ACDFuse 网络架构

论文: Adaptive Dual-Branch Fusion: Enhancing Infrared-Visible Image Integration via Cross-Modal Interaction

模块对照表:
- Encoder (共享分支): SERTE (Serial Enhanced Representation Transformer Encoder)
    - 包含: FreqStormer模块 (FreqMLP) + MKBlock (TransformerBlock)
    - 代码: EnhancedBaseFeatureExtraction + BaseFeatureExtraction + FreqMLP

- Encoder (特定分支): INN (Invertible Neural Network) 编码器
    - 代码: INNModule + DetailNode

- 自适应融合机制:
    - APCA: 自适应位置感知交叉注意力模块 (Adaptive Position-aware Cross-Attention)
    - HFL: 分层融合层 (Hierarchical Fusion Layer)

- Decoder: 包含 Bi-IRCSA (双向行间通道自注意力模块)
    - 代码: MultiScaleIRCSA + InterRowColSelfAttention
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ImprovementModule.kan import KANLinear
from ImprovementModule.FreMLP import Frequency_Domain
from einops import rearrange
from torch.nn import Softmax
from ImprovementModule.FADC import FADConv


# ============================================================================
# 工具函数
# ============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ============================================================================
# 基础组件
# ============================================================================

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    """层归一化包装器"""
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


import numbers


# ============================================================================
# 通道注意力
# ============================================================================

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# ============================================================================
# 频率域模块 (FreqStormer/MKBlock 的一部分)
# ============================================================================

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    def __init__(self, in_features, hidden_features=None, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)
        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)"""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ============================================================================
# 注意力机制
# ============================================================================

class AttentionBase(nn.Module):
    """基础注意力机制 (多头自注意力)"""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        return out


class Attention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA)"""
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


# ============================================================================
# Transformer 组件 (MKBlock)
# ============================================================================

class TransformerBlock(nn.Module):
    """Transformer块: 包含自注意力和前馈网络"""
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """重叠图像块嵌入 with 3x3 Conv"""
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


# ============================================================================
# MKBlock (Modulation Knowledge Block) - SERTE的核心组件
# ============================================================================

class MKBlock(nn.Module):
    """
    Modulation Knowledge Block (MKBlock)
    结合 KAN (Kolmogorov-Arnold Network) 进行特征调制
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor=1., qkv_bias=False):
        super(MKBlock, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = LayerNorm(dim, 'WithBias')
        # 使用 KANLinear 替代标准 MLP
        self.kan = KANLinear(
            in_features=dim,
            out_features=dim,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1]
        )

    def forward(self, x):
        # 注意力分支
        x = x + self.attn(self.norm1(x))

        # KAN特征调制
        b, c, h, w = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        x_reshaped = self.kan(x_reshaped)
        x = x_reshaped.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # 残差连接
        x = x + self.norm2(x)
        return x


# ============================================================================
# FreqStormer 模块 (频率域增强)
# ============================================================================

class FreqStormer(nn.Module):
    """
    FreqStormer: 频率域增强模块
    结合 Frequency_Domain (FreqMLP) 进行频率域特征增强
    """
    def __init__(self, dim, bias=False):
        super(FreqStormer, self).__init__()
        self.norm = LayerNorm(dim, 'WithBias')
        self.fre_mlp = Frequency_Domain(channels=dim)
        # 特征适配层
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        # 频率域增强
        x = x + self.fre_mlp(self.norm(x))
        # 特征适配
        x = x + self.feature_adapter(x)
        return x


# ============================================================================
# SERTE 编码器 (共享分支)
# Serial Enhanced Representation Transformer Encoder
# ============================================================================

class SERTEEncoder(nn.Module):
    """
    SERTE Encoder (共享分支编码器)
    包含:
    - 浅层特征提取: ShallowConvExtractor (基于FADC)
    - MKBlock堆叠: 多个TransformerBlock
    - FreqStormer: 频率域增强模块
    """
    def __init__(self,
                 inp_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(SERTEEncoder, self).__init__()

        # 浅层特征提取 (基于FADC)
        self.shallow_conv = ShallowConvExtractor(dim, dim)

        # 图像块嵌入层
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # MKBlock堆叠 (TransformerBlock序列)
        self.mkblock_layers = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])

        # MKBlock (KAN调制)
        self.mkblock = MKBlock(
            dim=dim,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor
        )

        # FreqStormer (频率域增强)
        self.freq_stormer = FreqStormer(dim=dim, bias=bias)

    def forward(self, inp_img):
        """前向传播: 返回 base_feature (共享特征)"""
        x = self.patch_embed(inp_img)
        x = self.shallow_conv(x)

        # MKBlock处理
        x = self.mkblock_layers(x)
        x = self.mkblock(x)

        # 频率域增强
        base_feature = self.freq_stormer(x)

        return base_feature, x


# ============================================================================
# INN 编码器 (特定分支)
# Invertible Neural Network for modality-specific features
# ============================================================================

class InvertedResidualBlock(nn.Module):
    """倒置残差块 (MobileNetV2)"""
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    """
    DetailNode: INN的单层节点
    包含三个变换函数: theta_phi, theta_rho, theta_eta
    用于特征间的耦合变换
    """
    def __init__(self):
        super(DetailNode, self).__init__()
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)  # 加性耦合
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)  # 仿射耦合
        return z1, z2


class INNEncoder(nn.Module):
    """
    INN Encoder (特定分支编码器)
    可逆神经网络编码器，以无损方式保留模态特定的边缘和纹理信息
    """
    def __init__(self, num_layers=3):
        super(INNEncoder, self).__init__()
        self.net = nn.Sequential(*[DetailNode() for _ in range(num_layers)])

    def forward(self, x):
        """前向传播: 返回 detail_feature (细节特征)"""
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


# ============================================================================
# 完整编码器 (双分支: 共享 + 特定)
# ============================================================================

class Encoder(nn.Module):
    """
    ACDFuse 完整编码器 (双分支架构)

    分支1 - 共享分支 (SERTE):
        建模模态共享一致性 (modality-shared consistency)
        包含: patch_embed -> shallow_conv -> MKBlock堆叠 -> FreqStormer

    分支2 - 特定分支 (INN):
        建模模态特定互补性 (modality-specific complementarity)
        包含: INNEncoder

    输出:
        base_feature: 共享基础特征
        detail_feature: 细节特征
        x: 浅层特征
    """
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder, self).__init__()

        # 共享分支编码器 (SERTE)
        self.serte_encoder = SERTEEncoder(
            inp_channels=inp_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

        # 特定分支编码器 (INN)
        self.inn_encoder = INNEncoder()

    def forward(self, inp_img):
        # 共享分支
        base_feature, shallow_feat = self.serte_encoder(inp_img)
        # 特定分支
        detail_feature = self.inn_encoder(shallow_feat)

        return base_feature, detail_feature, shallow_feat


# ============================================================================
# Bi-IRCSA (双向行间通道自注意力模块)
# ============================================================================

class InterRowColSelfAttention(nn.Module):
    """
    行列自注意力 (Inter Row/Column Self-Attention)
    用于Bi-IRCSA的水平或垂直方向的注意力计算
    """
    def __init__(self, in_dim, q_k_dim, axis='H'):
        super(InterRowColSelfAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.axis = axis

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.pos_embed = None
        self.softmax = Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.ca = ChannelAttention(in_dim)

    def _create_pos_embed(self, H, W):
        if self.axis == 'H':
            pos_embed = nn.Parameter(torch.zeros(1, self.q_k_dim, H, 1))
        elif self.axis == 'W':
            pos_embed = nn.Parameter(torch.zeros(1, self.q_k_dim, 1, W))
        else:
            raise ValueError("Axis must be one of 'H' or 'W'.")
        nn.init.xavier_uniform_(pos_embed)
        return pos_embed

    def forward(self, x, processed):
        B, C, H, W = x.size()

        if self.pos_embed is None or self.pos_embed.size()[2] != H or self.pos_embed.size()[3] != W:
            self.pos_embed = self._create_pos_embed(H, W).to(x.device)

        Q = self.query_conv(processed) + self.pos_embed
        K = self.key_conv(processed) + self.pos_embed
        V = self.value_conv(processed)
        scale = math.sqrt(self.q_k_dim)

        if self.axis == 'H':
            Q = Q.permute(0, 2, 3, 1).contiguous()
            Q = Q.view(B * W, H, self.q_k_dim)
            K = K.permute(0, 2, 3, 1).contiguous()
            K = K.view(B * W, H, self.q_k_dim).permute(0, 2, 1).contiguous()
            V = V.permute(0, 2, 3, 1).contiguous()
            V = V.view(B * W, H, self.in_dim)
            attn = torch.bmm(Q, K) / scale
            attn = self.softmax(attn)
            out = torch.bmm(attn, V)
            out = out.view(B, W, H, self.in_dim).permute(0, 3, 2, 1).contiguous()
        else:
            Q = Q.permute(0, 2, 3, 1).contiguous()
            Q = Q.view(B * H, W, self.q_k_dim)
            K = K.permute(0, 2, 3, 1).contiguous()
            K = K.view(B * H, W, self.q_k_dim).permute(0, 2, 1).contiguous()
            V = V.permute(0, 2, 3, 1).contiguous()
            V = V.view(B * H, W, self.in_dim)
            attn = torch.bmm(Q, K) / scale
            attn = self.softmax(attn)
            out = torch.bmm(attn, V)
            out = out.view(B, H, W, self.in_dim).permute(0, 3, 1, 2).contiguous()

        gamma = torch.sigmoid(self.gamma)
        out = gamma * out + (1 - gamma) * x
        ca_out = self.ca(out)
        out = out * ca_out

        return out


class BiIRCSA(nn.Module):
    """
    Bi-IRCSA: 双向行间通道自注意力模块
    Bidirectional Inter-Row/Column Self-Attention

    同时在水平和垂直方向进行注意力计算，增强空间-通道上下文建模
    """
    def __init__(self, in_dim, q_k_dim):
        super(BiIRCSA, self).__init__()
        self.horizontal_attn = InterRowColSelfAttention(in_dim, q_k_dim, axis='W')
        self.vertical_attn = InterRowColSelfAttention(in_dim, q_k_dim, axis='H')
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        h_out = self.horizontal_attn(x, x)
        v_out = self.vertical_attn(x, x)
        fused_output = self.fusion_weight * h_out + (1 - self.fusion_weight) * v_out
        return fused_output + x


# ============================================================================
# 解码器
# ============================================================================

class Decoder(nn.Module):
    """
    ACDFuse 解码器

    包含:
    - 通道融合层: 融合 base_feature 和 detail_feature
    - Bi-IRCSA: 双向行间通道自注意力
    - TransformerBlock堆叠: 特征精炼
    - 输出层: 生成融合图像
    """
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Decoder, self).__init__()

        # 通道融合层
        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        # Bi-IRCSA (双向行间通道自注意力)
        self.bi_ircsa = BiIRCSA(in_dim=dim, q_k_dim=dim // 4)

        # TransformerBlock堆叠
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[1])
        ])

        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature):
        # 融合 base 和 detail 特征
        if torch.equal(base_feature, detail_feature):
            out_enc_level0 = base_feature
        else:
            out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
            out_enc_level0 = self.reduce_channel(out_enc_level0)

        # Bi-IRCSA 增强
        spatial_attended = self.bi_ircsa(out_enc_level0)

        # TransformerBlock处理
        out_enc_level1 = self.encoder_level2(spatial_attended)

        # 残差连接生成输出
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)

        return self.sigmoid(out_enc_level1), out_enc_level0


# ============================================================================
# 浅层特征提取器 (FADC-based)
# ============================================================================

class ShallowConvExtractor(nn.Module):
    """
    浅层特征提取器
    基于 FADC (Frequency-Augmented Dual-branch Convolution) 模块
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            FADConv(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            FADConv(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# 工具函数
# ============================================================================

def count_parameters(model):
    """计算模型的参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ============================================================================
# SEBlock (Squeeze-and-Excitation)
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================================================
# FADN: Fusion-Enhanced Detection Neck Module (融合 → 检测)
# ============================================================================

class FADN(nn.Module):
    """
    FADN: 将融合特征 F_hfl 注入 YOLOv5 FPN 的残差增强模块

    双重职责:
    - 前向: F_hfl 经 SEBlock 通道筛选 + 残差注入 YOLOv5 FPN
    - 反向: 作为检测损失梯度回传至 ACDFuse 编码器的可微通道
    """
    def __init__(self, in_channels=64, fpn_channels=[128, 256, 512], reduction=16):
        super(FADN, self).__init__()
        self.fpn_channels = fpn_channels  # YOLOv5 FPN 各层通道数 [P3, P4, P5]

        # 红外分支对齐卷积 (参数不共享)
        self.align_ir = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fpn_channels[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_channels[i])
            ) for i in range(3)
        ])

        # 可见光分支对齐卷积 (参数不共享)
        self.align_vi = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fpn_channels[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_channels[i])
            ) for i in range(3)
        ])

        # SEBlock 通道重标定
        self.se_ir = nn.ModuleList([SEBlock(fpn_channels[i], reduction) for i in range(3)])
        self.se_vi = nn.ModuleList([SEBlock(fpn_channels[i], reduction) for i in range(3)])

        # 可学习融合系数
        self.beta_ir = nn.Parameter(torch.tensor(0.5))
        self.beta_vi = nn.Parameter(torch.tensor(0.5))

    def forward(self, f_hfl_ir, f_hfl_vi, fpn_features):
        """
        Args:
            f_hfl_ir: 红外分支融合特征
            f_hfl_vi: 可见光分支融合特征
            fpn_features: YOLOv5 FPN 的多尺度特征 [P3, P4, P5]
        Returns:
            增强后的 FPN 特征列表
        """
        enhanced_fpn = []
        for i, (f_pn, align_ir, align_vi, se_ir, se_vi) in enumerate(
                zip(fpn_features, self.align_ir, self.align_vi, self.se_ir, self.se_vi)):
            # 对齐融合特征到 FPN 分辨率
            f_hfl_ir_aligned = self._align_features(f_hfl_ir, f_pn, align_ir)
            f_hfl_vi_aligned = self._align_features(f_hfl_vi, f_pn, align_vi)

            # SE 通道筛选
            f_hfl_ir_se = se_ir(f_hfl_ir_aligned)
            f_hfl_vi_se = se_vi(f_hfl_vi_aligned)

            # 残差注入: F_new = F_fpn + β_ir * SE_ir(F_hfl_ir) + β_vi * SE_vi(F_hfl_vi)
            beta_ir = torch.sigmoid(self.beta_ir)
            beta_vi = torch.sigmoid(self.beta_vi)
            f_enhanced = f_pn + beta_ir * f_hfl_ir_se + beta_vi * f_hfl_vi_se

            enhanced_fpn.append(f_enhanced)

        return enhanced_fpn

    def _align_features(self, f_hfl, f_fpn, align_conv):
        """双线性插值对齐 + 1x1 卷积调整通道"""
        if f_hfl.shape[2:] != f_fpn.shape[2:]:
            f_hfl = F.interpolate(f_hfl, size=f_fpn.shape[2:], mode='bilinear', align_corners=False)
        return align_conv(f_hfl)


# ============================================================================
# TAFFM: Task-Aware Feature Feedback Module (检测 → 融合)
# ============================================================================

class TAFFM(nn.Module):
    """
    TAFFM: 用 YOLOv5 FPN 的多尺度特征 P3/P4/P5 调制 APCA 的注意力因子 α

    目的: 使融合在目标区域更精确，推理时双向互促持续生效
    """
    def __init__(self, fpn_channels=[128, 256, 512]):
        super(TAFFM, self).__init__()
        self.fpn_channels = fpn_channels

        # 各尺度 1x1 卷积压缩到单通道
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(fpn_channels[0], 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.conv_p4 = nn.Sequential(
            nn.Conv2d(fpn_channels[1], 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.conv_p5 = nn.Sequential(
            nn.Conv2d(fpn_channels[2], 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        # 可学习加权融合权重
        self.weight_p3 = nn.Parameter(torch.tensor(1.0))
        self.weight_p4 = nn.Parameter(torch.tensor(1.0))
        self.weight_p5 = nn.Parameter(torch.tensor(1.0))

    def forward(self, fpn_features, target_size):
        """
        Args:
            fpn_features: YOLOv5 FPN 的多尺度特征 [P3, P4, P5]
            target_size: 目标分辨率 (h, w)
        Returns:
            M_task: 任务感知掩码, 值域 [0,1], shape [B, 1, H, W]
        """
        p3, p4, p5 = fpn_features

        # 各尺度上采样到统一分辨率
        p3_up = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)

        # 压缩到单通道
        s_p3 = self.conv_p3(p3_up)
        s_p4 = self.conv_p4(p4_up)
        s_p5 = self.conv_p5(p5_up)

        # 加权融合
        w3 = torch.sigmoid(self.weight_p3)
        w4 = torch.sigmoid(self.weight_p4)
        w5 = torch.sigmoid(self.weight_p5)
        m_task = w3 * s_p3 + w4 * s_p4 + w5 * s_p5

        # Sigmoid 归一化到 [0,1]
        m_task = torch.sigmoid(m_task)

        return m_task


# ============================================================================
# APCA with TAFFM Modulation (带任务感知调制的 APCA)
# ============================================================================

class APCA_TAFFM(nn.Module):
    """
    APCA with TAFFM: 在原有 APCA 基础上增加任务感知注意力调制
    M_task 值域 [0,1]: 目标区域接近1(强化跨模态交互), 背景区域接近0(抑制冗余)
    """
    def __init__(self, dim, num_heads, bias=False):
        super(APCA_TAFFM, self).__init__()
        self.apca = APCA(dim=dim, num_heads=num_heads, bias=bias)
        self.dim = dim

    def forward(self, x, y, m_task=None):
        """
        Args:
            x: 红外特征
            y: 可见光特征
            m_task: 任务感知掩码 (可选), shape [B, 1, H, W]
        Returns:
            调制后的融合特征
        """
        if m_task is not None:
            # 调制 APCA 的自适应因子 alpha
            original_alpha = self.apca.alpha
            # alpha_new = alpha * m_task (逐元素调制, m_task 上采样到特征分辨率)
            b, c, h, w = x.shape
            if m_task.shape[2:] != (h, w):
                m_task = F.interpolate(m_task, size=(h, w), mode='bilinear', align_corners=False)
            # 调制后的 alpha
            modulated_alpha = original_alpha * m_task
            # 临时替换 alpha 进行前向传播
            self.apca.alpha = modulated_alpha.squeeze(1).mean()  # 用均值作为标量调制

        out = self.apca(x, y)

        # 恢复原始 alpha
        if m_task is not None:
            self.apca.alpha = original_alpha

        return out


# ============================================================================
# YOLOv5 检测头封装 (简化版, 用于联合训练)
# ============================================================================

class YOLOv5Head(nn.Module):
    """
    简化版 YOLOv5 检测头
    用于与 ACDFuse 联合训练, 接收 FPN 多尺度特征并输出检测结果
    """
    def __init__(self, num_classes=6, fpn_channels=[128, 256, 512], anchors_per_location=3):
        super(YOLOv5Head, self).__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.anchors_per_location = anchors_per_location

        # 每个 FPN 尺度的检测头
        self.det_heads = nn.ModuleList([
            nn.Sequential(
                # 特征提取
                nn.Conv2d(fpn_channels[i], fpn_channels[i], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels[i]),
                nn.LeakyReLU(0.1, inplace=True),
                # 类别预测: (num_classes + 5) * anchors_per_location
                nn.Conv2d(fpn_channels[i], (num_classes + 5) * anchors_per_location, kernel_size=1)
            ) for i in range(3)
        ])

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fpn_features):
        """
        Args:
            fpn_features: [P3, P4, P5] 三尺度特征列表
        Returns:
            检测输出列表
        """
        outputs = []
        for i, (feat, det_head) in enumerate(zip(fpn_features, self.det_heads)):
            outputs.append(det_head(feat))
        return outputs


# ============================================================================
# 兼容性别名 (保留旧名称以兼容已有代码)
# ============================================================================

# INNModule 是 INNEncoder 的别名
INNModule = INNEncoder

# MultiScaleIRCSA 是 BiIRCSA 的别名
MultiScaleIRCSA = BiIRCSA


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    height = 128
    width = 128

    # 创建模型实例
    modelE = Encoder().cuda()
    modelD = Decoder().cuda()

    # 测试前向传播
    x = torch.randn(1, 1, height, width).cuda()
    base_feature, detail_feature, shallow_feat = modelE(x)
    output, _ = modelD(x, base_feature, detail_feature)

    # 计算参数量
    total_params_E, trainable_params_E = count_parameters(modelE)
    total_params_D, trainable_params_D = count_parameters(modelD)
    total_params_all = total_params_E + total_params_D
    trainable_params_all = trainable_params_E + trainable_params_D

    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"\n模型参数统计:")
    print(f"编码器参数量: {total_params_E:,}")
    print(f"解码器参数量: {total_params_D:,}")
    print(f"总参数量: {total_params_all:,}")
    print(f"可训练参数量: {trainable_params_all:,}")
    print(f"参数量 (MB): {total_params_all * 4 / 1024 / 1024:.2f}")
