import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from .utils import SelectiveScan,flops_selective_scan_fn,\
    flops_selective_scan_ref,print_jit_input_names, Mlp, CrossScan,CrossMerge,\
    selective_scan_flatten, SEModule, ConvFFN, x_selective_scan


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
    force_fp32=True,
    **kwargs,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
    
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    )

    ys = ys.view(B, K, -1, H, W)
    y: torch.Tensor = CrossMerge.apply(ys) # (B, C, H, W)
    y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops

# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x



class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        forward_type="v2",
        # ======================
        **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.d_inner = d_inner
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)
        if kwargs.get('sscore_type','None') != 'None':
            ms_stage,current_stage = kwargs.get('ms_stage'),kwargs.get('current_layer')
            if current_stage not in ms_stage:
                kwargs['sscore_type'] = 'None'

        if kwargs.get('sscore_type','None') in ['multiscale_4scan_12']:
            forward_type = "multiscale_ssm"
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        if kwargs.get('sscore_type','None') in ['multiscale_4scan_12']:
            self.K = 1 + kwargs.get('ms_split')[0]

        self.K2 = self.K if forward_type not in ["share_a"] else 1

        if kwargs.get('add_se',False):
            self.se = SEModule(d_expand, reduction=8)

        if kwargs.get('ms_fusion', None) == None:
            if kwargs.get('upsample',None) == 'interpolate':
                pass
            elif kwargs.get('upsample',None) == 'conv':
                if kwargs.get('current_layer', 0) == 3:
                    self.upsample = nn.ConvTranspose2d(
                        in_channels=d_expand,
                        out_channels=d_expand,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=d_expand,
                        bias=conv_bias,
                    )
                else:
                    self.upsample = nn.ConvTranspose2d(
                        in_channels=d_expand,
                        out_channels=d_expand,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        groups=d_expand,
                        bias=conv_bias,
                    )

        # forward_type =======================================
        self.forward_core = dict(
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            flatten_ssm=self.forward_core_flatten,
            multiscale_ssm=self.forward_core_multiscale,
        ).get(forward_type, self.forward_corev2)


        # in proj =======================================

        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            stride = 1
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                stride=stride,
                **factory_kwargs,
            )# branch 0, convert B, C, H, W to B, C, H, W
        if kwargs.get('sscore_type','None') in ['multiscale_4scan_12']:
            if kwargs.get('add_conv', True):
                b1_stride = 2
                self.conv2d_b1 = nn.Conv2d(
                    in_channels=d_expand,
                    out_channels=d_expand,
                    groups=d_expand,
                    bias=conv_bias,
                    kernel_size=7,
                    stride=b1_stride,
                    padding=3,
                    **factory_kwargs,
                ) #bracnh 1, convert B, C, H, W to B, C, H//4, W//4
            if kwargs.get('sep_norm', False):
                self.out_norm0 = self.out_norm
                self.out_norm1 = nn.LayerNorm(d_inner)

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # other kwargs =======================================
        self.kwargs = kwargs
        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        self.debug = False

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if self.debug: debug_rec = []
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = x_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
            **self.kwargs,
        )
        x, debug_rec = x[0], x[1]
        if self.ssm_low_rank:
            x = self.out_rank(x)
        if self.debug:
            return x, debug_rec
        return  x

    def forward_core_flatten(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 2, 1).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = x.transpose(1, 2).contiguous() # back to (B, L, C)
        x = selective_scan_flatten(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
            **self.kwargs,
        ) # (B, L, C)
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x
    def forward_core_multiscale(self, xs: list, nrows=-1, channel_first=False):
        nrows = 1
        ys, debug_rec = [], []
        for i,x in enumerate(xs):
            if not channel_first:
                x = x.permute(0, 2, 1).contiguous()
            if self.ssm_low_rank:
                x = self.in_rank(x)
            if self.kwargs.get('sep_norm', False):
                norm_name = getattr(self, "out_norm" + str(i), nn.LayerNorm(self.d_inner))
            else:
                norm_name = getattr(self, "out_norm", None)
            if i == 0:
                proj_weight = self.x_proj_weight[[i]]
                dt_projs_weight = self.dt_projs_weight[[i]]
                dt_projs_bias = self.dt_projs_bias[[i]]
                A_logs = self.A_logs[i*self.d_inner:(i+1)*self.d_inner]
                Ds = self.Ds[i*self.d_inner:(i+1)*self.d_inner]
            else:
                proj_weight = self.x_proj_weight[i:]
                dt_projs_weight = self.dt_projs_weight[i:]
                dt_projs_bias = self.dt_projs_bias[i:]
                A_logs = self.A_logs[i*self.d_inner:]
                Ds = self.Ds[i*self.d_inner:]
            #if not debug  mode, remove x_rec
            x, debug = x_selective_scan(
                x, proj_weight, None, dt_projs_weight, dt_projs_bias,
                A_logs, Ds,
                norm_name,
                nrows=nrows, delta_softplus=True, force_fp32=self.training,
                **self.kwargs,
            )

            if self.ssm_low_rank:
                x = self.out_rank(x) # (B, L, C)
            ys.append(x)
            debug_rec.append(debug)
        if self.debug:
            return ys, debug_rec
        return ys

    def forward(self, x: torch.Tensor,h_tokens=None,w_tokens=None, **kwargs):

        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            b, h, w, d = x.shape
            if not self.disable_z_act:
                z = self.act(z) # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            x_b1 = x
            x = self.act(self.conv2d(x)) # (b, d, h, w)
            if self.kwargs.get('sscore_type','None') in ['multiscale_4scan_12']:
                if self.kwargs.get('add_conv', True):
                    x_b1 = self.act(self.conv2d_b1(x_b1)) # (b, d, h//4, w//4)
                h_b1, w_b1 = x_b1.shape[2:]
                #reverse horizontal scan
                x_hori_r = x_b1.flatten(2).flip(-1)
                #vertical scan
                x_vert = x_b1.transpose(2, 3).flatten(2).contiguous()
                #reverse vertical scan
                x_vert_r = x_b1.transpose(2, 3).flatten(2).flip(-1).contiguous()
                splits = self.kwargs.get('ms_split')[1]
                if splits == 3:
                    x_b1 = torch.cat([x_hori_r, x_vert, x_vert_r], dim=2) # (b, d, h//2*w//2*3)
                    x = x.flatten(2) # (b, d, h*w)
                elif splits == 2:
                    x_b1 = torch.cat([x_hori_r, x_vert_r], dim=2)
                elif splits == 1:
                    x_b1 = x_vert_r
                x = [x_b1, x]
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        if self.debug:
            y, debug_rec = self.forward_core(x, channel_first=(self.d_conv > 1))
        else:
            y = self.forward_core(x, channel_first=(self.d_conv > 1))

        if self.kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            y_b0, y_b1 = y[1], y[0]#  (b, h//4*w//4*3, d)
            if splits == 3:
                y_hori_r = y_b1[:,:h_b1*w_b1].flip(-2).view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
                y_vert = y_b1[:,h_b1*w_b1:h_b1*w_b1*2].view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
                y_vert_r = y_b1[:,h_b1*w_b1*2:].flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
                y_b1 = y_hori_r + y_vert + y_vert_r
            elif splits==2:
                y_hori_r = y_b1[:,:h_b1*w_b1].flip(-2).view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
                y_vert_r = y_b1[:,h_b1*w_b1:].flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
                y_b1 = y_hori_r + y_vert_r
            elif splits==1:
                y_vert_r = y_b1.flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
                y_b1 = y_vert_r
            y_b1 = F.interpolate(y_b1, size=(h, w), mode='bilinear', align_corners=False)
            y_b1 = y_b1.permute(0, 2, 3, 1).contiguous()
            y = y_b0.view(b, h, w, -1) + y_b1

        if getattr(self, "__DEBUG__", False):
            if self.kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
                ys_b1 = debug_rec[0]['ys'].view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b*3, -1, h_b1, w_b1)
                ys_b1 = F.interpolate(ys_b1, size=(h, w), mode='bilinear', align_corners=False) # (b*3, d, h, w)
                ys_b1 = ys_b1.view(b, 3, -1, h, w).contiguous() # (b, 3, d, h, w)
                temp = ys_b1[:, 0].clone()
                ys_b1[:, 0] = ys_b1[:, 1]
                ys_b1[:, 1] = temp
                ys_b0 = debug_rec[1]['ys'].view(b, 1, -1, h, w).contiguous() # (b, 1, d, h, w)
                ys = torch.cat([ys_b0, ys_b1], dim=1) # (b, 4, d, h, w)

                xs_b1 = debug_rec[0]['xs'].view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b*3, -1, h_b1, w_b1)
                xs_b1 = F.interpolate(xs_b1, size=(h, w), mode='bilinear', align_corners=False)
                xs_b1 = xs_b1.view(b, 3, -1, h, w).contiguous()
                temp = xs_b1[:, 0].clone()
                xs_b1[:, 0] = xs_b1[:, 1]
                xs_b1[:, 1] = temp
                xs_b0 = debug_rec[1]['xs'].view(b, 1, -1, h, w).contiguous()
                xs = torch.cat([xs_b0, xs_b1], dim=1).view(b, -1, h*w) # (b, 4, d, h, w)

                A_logs_b1, dts_b1, bs_b1, cs_b1, ds_b1, delta_bias_b1 = debug_rec[0]['As'], debug_rec[0]['dts'], debug_rec[0]['Bs'], debug_rec[0]['Cs'], debug_rec[0]['Ds'], debug_rec[0]['delta_bias']
                A_logs_b0, dts_b0, bs_b0, cs_b0, ds_b0, delta_bias_b0 = debug_rec[1]['As'], debug_rec[1]['dts'], debug_rec[1]['Bs'], debug_rec[1]['Cs'], debug_rec[1]['Ds'], debug_rec[1]['delta_bias']
                A_logs = torch.cat([A_logs_b0, A_logs_b1.repeat(3, 1)], dim=0)
                delta_bias = torch.cat([delta_bias_b0, delta_bias_b1.repeat(3)], dim=0)
                ds = torch.cat([ds_b0, ds_b1.repeat(3)], dim=0)
                # dts, bs, cs correspond to v,k,q in transformer
                dts_b1 = dts_b1.view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b*3, -1, h_b1, w_b1)
                dts_b1 = F.interpolate(dts_b1, size=(h, w), mode='bilinear', align_corners=False)
                dts_b1 = dts_b1.view(b, 3, -1, h*w)
                temp = dts_b1[:, 0].clone()
                dts_b1[:, 0] = dts_b1[:, 1]
                dts_b1[:, 1] = temp

                dts_b1 = dts_b1.view(b, -1, h*w).contiguous()
                dts = torch.cat([dts_b0, dts_b1], dim=1).view(b, -1, h*w) # (b, 4, d, h*w)

                bs_b1 = bs_b1.view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b*3, -1, h_b1, w_b1)
                bs_b1 = F.interpolate(bs_b1, size=(h, w), mode='bilinear', align_corners=False)
                bs_b1 = bs_b1.view(b, 3, -1, h * w)
                temp = bs_b1[:, 0].clone()
                bs_b1[:, 0] = bs_b1[:, 1]
                bs_b1[:, 1] = temp
                bs = torch.cat([bs_b0, bs_b1], dim=1) # (b, 4, d, h*w)

                cs_b1 = cs_b1.view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b*3, -1, h_b1, w_b1)
                cs_b1 = F.interpolate(cs_b1, size=(h, w), mode='bilinear', align_corners=False)
                cs_b1 = cs_b1.view(b, 3, -1, h*w)
                temp = cs_b1[:, 0].clone()
                cs_b1[:, 0] = cs_b1[:, 1]
                cs_b1[:, 1] = temp
                cs = torch.cat([cs_b0, cs_b1], dim=1) # (b, 4, d, h*w)
            else:
                xs, ys = debug_rec['xs'], debug_rec['ys']
                A_logs, dts, bs, cs, ds, delta_bias = debug_rec['As'], debug_rec['dts'], debug_rec['Bs'], debug_rec['Cs'], debug_rec['Ds'], debug_rec['delta_bias']



            setattr(self, "__data__", dict(
                A_logs=A_logs, Bs=bs, Cs=cs, Ds=ds,
                us=xs, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))

        if self.kwargs.get('add_se',False):
            y = y.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
            y = self.se(y)
            y = y.permute(0, 2, 3, 1).contiguous()
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)



class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                simple_init=ssm_simple_init,
                # ==========================
                forward_type=forward_type,
                **kwargs,
            )

        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)
        if kwargs.get('convFFN', False):
            ffn_drop = kwargs.get('ffn_dropout', 0.2)
            self.convFFN = ConvFFN(hidden_dim, expansion=2, drop=ffn_drop)
            self.norm2 = norm_layer(hidden_dim)
        self.kwargs = kwargs

    def _forward(self, input: torch.Tensor, h_tokens=None, w_tokens=None):
        if self.ssm_branch:
                x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        if self.kwargs.get('convFFN', False):
            x = x + self.drop_path(self.convFFN(self.norm2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous())

        return x

    def forward(self, input: torch.Tensor,h_tokens=None, w_tokens=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input,h_tokens=None, w_tokens=None)
        else:
            return self._forward(input,h_tokens=None, w_tokens=None)


class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_simple_init=False,
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN",
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        if type(norm_layer)==str and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            kwargs.update({'current_layer': i_layer})
            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                **kwargs,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))
        self.kwargs = kwargs
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {}

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()), 
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )


    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    def _make_downsample_1D(self, dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 2, 1),
            nn.Conv1d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_simple_init=False,
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                **kwargs,
            ))
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor, h_tokens=None, w_tokens=None):
        # from .vis import visualize_batch
        # img = x
        x = self.patch_embed(x)
        for layer in self.layers:
            for block in layer.blocks:
                x = block(x)
            x = layer.downsample(x)
        #calculate the similarity between the center patch and other patches, x: (B, H, W, C)
        # visualize_batch(img, block_outputs)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer=nn.LayerNorm, **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs


# ==================================================
def check_vssm_equals_vmambadp():
    try:
        from _ignore.vmamba.vmamba_bak1 import VMamba2Dp
        from _ignore.vmamba.vmamba_pub import VSSM
    except:
        print("original VSSM and VMamba2Dp not found.", flush=True)
        return 

    # test 1 True =================================
    torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
    oldvss = VMamba2Dp(depths=[2,2,6,2]).half().cuda()
    newvss = VSSM(depths=[2,2,6,2]).half().cuda()
    newvss.load_state_dict(oldvss.state_dict())
    input = torch.randn((12, 3, 224, 224)).half().cuda()
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward_backbone(input)
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward_backbone(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward(input)
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    # test 2 True ==========================================
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    oldvss = VMamba2Dp(depths=[2,2,6,2]).cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    newvss = VSSM(depths=[2,2,6,2]).cuda()

    miss_align = 0
    for k, v in oldvss.state_dict().items(): 
        same = (oldvss.state_dict()[k] == newvss.state_dict()[k]).all()
        if not same:
            print(k, same)
            miss_align += 1
    print("init miss align", miss_align) # init miss align 0


def check_vssm1_equals_vssm(forward_type="v0"):
    try:
        from _ignore.vmamba.vmamba_pub import VSSM as VSSM0
    except:
        print("original VSSM and VMamba2Dp not found.", flush=True)
        return

    class VSSM_(VSSM):
        @staticmethod
        def _make_layer(*args, **kwargs):
            layer = VSSM._make_layer(*args, **kwargs)
            dim = kwargs.get("dim", None)
            norm_layer = kwargs.get("norm_layer", None)
            downsample = kwargs.get("downsample", None)
            downsample_1D = kwargs.get("downsample_1D", None)
            blocks = layer.blocks
        
            if True: # is this really applied? Yes, but been overriden later in VSSM!
                def _init_weights(module: nn.Module):
                    for name, p in module.named_parameters():
                        if name in ["out_proj.weight"]:
                            p = p.clone().detach_() # fake init, just to keep the seed ....
                            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                blks = nn.Sequential(*copy.deepcopy(blocks))
                blks.apply(_init_weights)

            downsample = PatchMerging2D(dim, 2*dim, norm_layer=norm_layer) if downsample is None else nn.Identity()
            #downsample_1D = (dim, 2*dim, norm_layer=norm_layer) if downsample_1D is None else nn.Identity()
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks,),
                downsample=downsample,
            ))

        def forward_backbone(self, x):
            x = self.patch_embed(x)
            for l in self.layers:
                x = l(x)
            return x

        def forward1(self, x: torch.Tensor):
            x = self.patch_embed(x)
            for layer in self.layers:
                x = layer(x)
            x = self.classifier.norm(x)
            # here: whether has contiguous would differ
            x = self.classifier.avgpool(x.permute(0, 3, 1, 2).contiguous()).flatten(1)
            x = self.classifier.head(x)
            return x

    # only has initial difference 
    VSSM1 = partial(VSSM, downsample_version="v1", patchembed_version="v1", mlp_ratio=0.0, ssm_ratio=2.0, ssm_rank_ratio=2.0, forward_type=forward_type)
    VSSM.forward_backbone = VSSM_.forward_backbone 
    VSSM.forward1 = VSSM_.forward1
    # expected to be all the same 
    VSSM1 = partial(VSSM_, downsample_version="none", patchembed_version="v1", mlp_ratio=0.0, ssm_ratio=2.0, ssm_rank_ratio=2.0, forward_type=forward_type)

    # test 1 True =================================
    torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
    oldvss = VSSM0(depths=[2,2,6,2]).half().cuda()
    newvss = VSSM1(depths=[2,2,6,2]).half().cuda()
    newvss.load_state_dict(oldvss.state_dict())
    input = torch.randn((12, 3, 224, 224)).half().cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward_backbone(input)
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward_backbone(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward(input)
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward1(input)
    print((y1 -y2).abs().sum()) # tensor(2.5988e-05, device='cuda:0', grad_fn=<SumBackward0>)
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y3 = newvss.forward(input)
    print((y1 -y3).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    # test 2 True ==========================================
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    oldvss = VSSM0(depths=[2,2,6,2]).cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    newvss = VSSM1(depths=[2,2,6,2]).cuda()

    miss_align = 0
    oldvss2new = copy.deepcopy(newvss)
    oldvss2new.load_state_dict(oldvss.state_dict())
    for k, v in oldvss2new.state_dict().items(): 
        same = (oldvss2new.state_dict()[k] == newvss.state_dict()[k]).all()
        if not same:
            print(k, same)
            miss_align += 1
    print("init miss align", miss_align) # init miss align 0


def check_profile():
    vss = VSSM(depths=[1], dims=1024).half().cuda()
    input = torch.randn((128, 3, 56, 56)).half().cuda()
    torch.cuda.manual_seed(0)

    self = vss
    blk = self.layers[0].blocks[0]
    ln_1 = blk.ln_1
    self_attention = blk.self_attention
    selfa = self_attention
    drop_path = blk.drop_path
    input = self.patch_embed(input).detach()

    def trace_handler(prof: torch.profiler.profile):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        # print(prof.export_chrome_trace("./tracev1.json"))

    with torch.cuda.amp.autocast():
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=True, with_stack=True) as prof:
        with torch.profiler.profile(
            with_modules=True,
            with_stack=True,
            profile_memory=True,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2,
                repeat=1),
            on_trace_ready=trace_handler
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
            ) as prof:
                for iter in range(1000):
                    x = input
                    # with torch.autograd.profiler.record_function("patch_embed"):
                    #     x = self.patch_embed(x)
                    
                    B, H, W, C = x.shape
                    ori = x

                    with torch.autograd.profiler.record_function("VSSBlock.ln_1"):
                        x = ln_1(x)

                    with torch.autograd.profiler.record_function("SS2D.inproj"):
                        xz = selfa.in_proj(x)
                        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                        x = x.permute(0, 3, 1, 2).contiguous()

                    with torch.autograd.profiler.record_function("SS2D.dwconv2d"):
                        x = selfa.act(selfa.conv2d(x)) # (b, d, h, w)
                        # x = self.act(x) # (b, d, h, w)
                    
                    with torch.autograd.profiler.record_function("SS2D.foreward_core"):
                        # y = selfa.forward_corev2(x)
                        # y = selfa.forward_corev3(x)
                        y = selfa.forward_corev1(x)
                        # y = selfa.forward_corev1(x)
                    
                    with torch.autograd.profiler.record_function("SS2D.transpose"):
                        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
                        y = selfa.out_norm(y)
                        y = y * F.silu(z)
                    
                    with torch.autograd.profiler.record_function("SS2D.out_proj"):
                        out = selfa.out_proj(y)
                        if selfa.dropout is not None:
                            out = selfa.dropout(out)

                    with torch.autograd.profiler.record_function("SS2D.out"):
                        x = ori + drop_path(out)

                    with torch.autograd.profiler.record_function("backward"):
                        x.sum().backward()

                    prof.step()


def load22kto1k():
    if False:
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = model.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    # bicubic interpolate relative_position_bias_table if not match
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                        mode='bicubic')
                    state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

        # bicubic interpolate absolute_pos_embed if not match
        absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
        for k in absolute_pos_embed_keys:
            # dpe
            absolute_pos_embed_pretrained = state_dict[k]
            absolute_pos_embed_current = model.state_dict()[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C1:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                    absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                        absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                    state_dict[k] = absolute_pos_embed_pretrained_resized

        # check classifier, if not match, then re-init classifier to zero
        head_bias_pretrained = state_dict['head.bias']
        Nc1 = head_bias_pretrained.shape[0]
        Nc2 = model.head.bias.shape[0]
        if (Nc1 != Nc2):
            if Nc1 == 21841 and Nc2 == 1000:
                logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
                map22kto1k_path = f'data/map22kto1k.txt'
                with open(map22kto1k_path) as f:
                    map22kto1k = f.readlines()
                map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
                state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
            else:
                torch.nn.init.constant_(model.head.bias, 0.)
                torch.nn.init.constant_(model.head.weight, 0.)
                del state_dict['head.weight']
                del state_dict['head.bias']
                logger.warning(f"Error in loading classifier head, re-init classifier head to 0")


if __name__ == "__main__":
    check_vssm_equals_vmambadp()
    check_vssm1_equals_vssm(forward_type="v0")
    check_vssm1_equals_vssm(forward_type="v0_seq")
    check_vssm1_equals_vssm(forward_type="v2")
    print(VSSM(forward_type="v0").flops())
    print(VSSM(forward_type="v2").flops())
    print(VSSM(forward_type="v2nozact").flops())

    


