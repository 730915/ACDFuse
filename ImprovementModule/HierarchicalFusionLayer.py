import torch
import torch.nn as nn

from ImprovementModule.APCA import APCA

class HierarchicalFusionLayer(nn.Module):
    def __init__(self, dim, num_heads, num_layers=3, fusion_mode='sum'):
        """
        Hierarchical Fusion Layer

        Args:
            dim (int): number of channels
            num_heads (int): number of attention heads
            num_layers (int): number of LCA layers
            fusion_mode (str): how to fuse final base/detail features
                - 'sum': fused = base + detail
                - 'concat': fused = concat(base, detail) then 1x1 conv
        """
        super().__init__()
        self.num_layers = num_layers
        self.fusion_mode = fusion_mode.lower()
        
        self.layers_d2b = nn.ModuleList([
            APCA(dim, num_heads) for _ in range(num_layers)
        ])
        self.layers_b2d = nn.ModuleList([
            APCA(dim, num_heads) for _ in range(num_layers)
        ])
        
        if self.fusion_mode == 'concat':
            self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

    def forward(self, base_feat, detail_feat):
        """
        base_feat: Tensor of shape [B, C, H, W]
        detail_feat: Tensor of shape [B, C, H, W]
        """
        out_b = base_feat
        out_d = detail_feat

        for layer_b2d, layer_d2b in zip(self.layers_b2d, self.layers_d2b):
            # base -> detail
            out_d = layer_b2d(out_d, out_b)
            
            # detail -> base
            out_b = layer_d2b(out_b, out_d)

        # 最终融合
        if self.fusion_mode == 'sum':
            fused = out_b + out_d
        elif self.fusion_mode == 'concat':
            fused = torch.cat([out_b, out_d], dim=1)
            fused = self.fusion_conv(fused)
        else:
            raise NotImplementedError(f"Unsupported fusion_mode: {self.fusion_mode}")

        return fused
