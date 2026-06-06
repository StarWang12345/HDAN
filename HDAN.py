import torch.nn as nn
import ops as ops
import torch
import torch.nn.functional as F

class ACAG(nn.Module):
    def __init__(self, channels, wn):
        super().__init__()
        self.global_processor = ops.DEMSB(wn, in_ch=channels, out_ch=channels)
        self.local_processor = nn.Sequential(
            ops.OptimizedResBlock(channels),
            ops.OptimizedLocalAgg(channels)
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            wn(nn.Conv2d(channels * 2, channels, 1)),
            nn.ReLU(inplace=True),
            wn(nn.Conv2d(channels, channels * 2, 1)),
            nn.Tanh()
        )
        self.final_conv = wn(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        shortcut = x
        local_feat = self.local_processor(x)
        global_feat = self.global_processor(x)
        feats = torch.cat([local_feat, global_feat], dim=1)
        weights = self.att(feats)
        w_local, w_global = torch.chunk(weights, 2, dim=1)
        fused = local_feat * (1+w_local) + global_feat * (1+w_global)
        out = self.final_conv(fused)
        return out + shortcut
    
class ACAG_L(nn.Module):
    def __init__(self, channels, wn):
        super().__init__()
        self.local_processor = nn.Sequential(
            ops.OptimizedResBlock(channels),
            ops.OptimizedLocalAgg(channels)
        )
        self.final_conv = wn(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        shortcut = x
        local_feat = self.local_processor(x)
        out = self.final_conv(local_feat)

        return out + shortcut
class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        scale = kwargs.get("scale")
        group = kwargs.get("group", 4)
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))
        # HDAN like this
        self.blocks = nn.ModuleList([
            ACAG(64, wn=wn) for _ in range(7)] + [
            ACAG_L(64, wn=wn)]
            )
        # HDAN-L like this
        # self.blocks = nn.ModuleList([
        #     ACAG(64, wn=wn) for _ in range(5)] + [
        #     ACAG_L(64, wn=wn)]
        #     )        
        # HDAN like this
        self.agg_fusion = wn(nn.Conv2d(64 * 8, 64, 1, 1, 0))
        # HDAN-L like this
        # self.agg_fusion = wn(nn.Conv2d(64 * 6, 64, 1, 1, 0))   
        #     
        self.reduction_0 = ops.BasicConv2d(wn, 64 * 2, 64, 1, 1, 0)
        self.reduction_1 = ops.BasicConv2d(wn, 64 * 4, 64, 1, 1, 0)
        self.reduction_2 = ops.BasicConv2d(wn, 64 * 3, 64, 1, 1, 0)
        self.upsample = ops.UpsampleBlock(64, scale=scale, multi_scale=False, wn=wn, group=group)
        self.exit1 = wn(nn.Conv2d(64, 3, 3, 1, 1))
        self.skip_conv = wn(nn.Conv2d(3, 64, 1, 1, 0))
    def forward(self, x, scale):
        x = self.sub_mean(x)
        res = x
        x_in = self.entry_1(x)
        feats = []
        out = x_in
        for block in self.blocks:
            out = block(out)
            feats.append(out)
        out = torch.cat(feats, dim=1)
        out = self.agg_fusion(out)
        out = out + x_in
        out = self.upsample(out, scale=scale)
        out = self.exit1(out)
        skip = F.interpolate(res, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)
        out = skip + out
        out = self.add_mean(out)

        return out