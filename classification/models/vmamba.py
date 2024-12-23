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
#from mamba_ssm import Mamba2
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchvision.models import VisionTransformer

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
    from .csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F
    from .csms6s import CrossScan, CrossMerge
    from .csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction
    from .csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
    from .csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit
    from .csms6s import SEModule,ConvFFN,Linear2d,LayerNorm2d,PatchMerging2D,Permute,Mlp,gMlp,SoftmaxSpatial,SS2Dv0,SS2Dv3
    from .csms6s import axis_scan2d,axis_merge2d,Mamba2, mamba_chunk_scan_combined_fn, mamba_chunk_scan_combined_flop_jit
    from .linear_attn import GatedLinearAttention,ConvLayer,Stem,PatchMerging
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
    from csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F
    from csms6s import CrossScan, CrossMerge, CrossScan3, CrossMerge3
    from csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction
    from csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
    from csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit
    from csms6s import SEModule,ConvFFN,Linear2d,LayerNorm2d,PatchMerging2D,Permute,Mlp,gMlp,SoftmaxSpatial,SS2Dv0,SS2Dv3
    from csms6s import axis_scan2d,axis_merge2d,Mamba2, mamba_chunk_scan_combined_fn, mamba_chunk_scan_combined_flop_jit
    from linear_attn import GatedLinearAttention,ConvLayer,Stem,PatchMerging




# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

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
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
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


# support: v0, v0seq
# support: v01-v05; v051d,v052d,v052dc; 
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
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
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.kwargs = kwargs
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_cnorm:
            self.out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba), # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
            ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
            ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, cascade2d=True),
            v05notriton=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
            ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
            ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
            # -------------------------------
            vms = partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=None, CrossMerge=None),
            axis = partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=None, CrossMerge=None),
        )
        k_group = 4
        if self.kwargs.get('add_se', False):
            self.se = SEModule(d_inner, reduction=8)
        if forward_type == 'vms':
            k_group = 2
            self.conv2d_b1 = nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    **factory_kwargs,
                )
            self.full_res_index = kwargs.get('full_res_index', 1)
        elif forward_type == 'axis':
            if kwargs.get('current_stage',0) in kwargs.get('axis_stage'):
                k_group = 6
            else:
                forward_type = "v05"
        self.forward_type = forward_type
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((k_group, d_inner, dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=SelectiveScanOflex,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        cascade2d=False,
        # ==============================
        x_b1 = None,
        **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        def get_merged_xs(x, x_full, full_res_index):
            x_hori = x.flatten(2)
            x_hori_r = x.flatten(2).flip(-1)
            x_vert = x.transpose(2, 3).flatten(2).contiguous()
            x_vert_r = x.transpose(2, 3).flatten(2).flip(-1).contiguous()

            if full_res_index == 1: # horizontal scan is full resolution
                return torch.cat([x_hori_r, x_vert, x_vert_r], dim=2).contiguous(), x_full.flatten(2)
            elif full_res_index == 2: # horizontal reverse scan is full resolution
                return torch.cat([x_hori, x_vert, x_vert_r], dim=2).contiguous(), x_full.flatten(2).flip(-1)
            elif full_res_index == 3: # vertical scan is full resolution
                return torch.cat([x_hori, x_hori_r, x_vert_r], dim=2).contiguous(), x_full.transpose(2, 3).flatten(2)
            elif full_res_index == 4: # vertical reverse scan is full resolution
                return torch.cat([x_hori, x_hori_r, x_vert], dim=2).contiguous(), x_full.transpose(2, 3).flatten(2).flip(-1)

        def get_splited_xs(x, x_full, full_res_index,H_b1,W_b1):
            x_parts = x.chunk(3, dim=2)
            H,W = x_full.shape[-2:]
            if full_res_index == 1: # horizontal scan is full resolution
                x_hori_r = x_parts[0].flip(-1).view(B, -1, H_b1, W_b1)
                x_vert = x_parts[1].view(B, -1, W_b1, H_b1).transpose(2, 3)
                x_vert_r = x_parts[2].flip(-1).view(B, -1, W_b1, H_b1).transpose(2, 3)
                return x_hori_r + x_vert + x_vert_r, x_full
            elif full_res_index == 2: # horizontal reverse scan is full resolution
                x_hori = x_parts[0].view(B, -1, H_b1, W_b1)
                x_vert = x_parts[1].view(B, -1, W_b1, H_b1).transpose(2, 3)
                x_vert_r = x_parts[2].flip(-1).view(B, -1, W_b1, H_b1).transpose(2, 3)
                return x_hori + x_vert + x_vert_r, x_full.flatten(2).flip(-1).view(B, -1, H, W).contiguous()
            elif full_res_index == 3: # vertical scan is full resolution
                x_hori = x_parts[0].view(B, -1, H_b1, W_b1)
                x_hori_r = x_parts[1].flip(-1).view(B, -1, H_b1, W_b1)
                x_vert_r = x_parts[1].flip(-1).view(B, -1, W_b1, H_b1).transpose(2, 3).contiguous()
                return x_hori_r + x_hori + x_vert_r, x_full.transpose(2, 3).contiguous()
            elif full_res_index == 4: # vertical reverse scan is full resolution
                x_hori = x_parts[0].view(B, -1, H_b1, W_b1)
                x_hori_r = x_parts[1].flip(-1).view(B, -1, H_b1, W_b1)
                x_vert = x_parts[1].view(B, -1, W_b1, H_b1).transpose(2, 3)
                return x_hori_r + x_hori + x_vert, x_full.flatten(2).flip(-1).view(B, -1, W, H).transpose(2, 3).contiguous()

        if self.forward_type == 'vms':
            B, D, H_b1, W_b1 = x_b1.shape # (B, D, H, W)
            L_b1 = H_b1 * W_b1 * 3
            x_b1,x_b0 = get_merged_xs(x_b1, x, self.full_res_index)
            # x_hori_r = x_b1.flatten(2).flip(-1)
            # x_vert = x_b1.transpose(2, 3).flatten(2).contiguous()
            # x_vert_r = x_b1.transpose(2, 3).flatten(2).flip(-1).contiguous()
            # x_b1 = torch.cat([x_hori_r, x_vert, x_vert_r], dim=2).contiguous()  # (b, d, h//2*w//2*3)

            xs_b1 = x_b1.unsqueeze(1) # (B, C,  h//2*w//2*3) -> (B, 1, C, h//2*w//2*3)
            xs_b0 = x_b0.unsqueeze(0) # (B, C, H*W) -> (B, 1, C, H*W)
            x_proj_weight_b0 = x_proj_weight.view(2, -1, D)[[0]].unsqueeze(0).contiguous() # (K, N, inner) -> (1, N, inner)
            x_proj_weight_b1 = x_proj_weight.view(2, -1, D)[[1]].unsqueeze(0).contiguous() # (K, N, inner) -> (1, N, inner)
            dt_projs_weight_b0 = dt_projs_weight.view(2, D, -1)[[0]].unsqueeze(0).contiguous() # (K, inner, rank) -> (1, inner, rank)
            dt_projs_weight_b1 = dt_projs_weight.view(2, D, -1)[[1]].unsqueeze(0).contiguous() # (K, inner, rank) -> (1, inner, rank)
            dt_projs_bias_b0 = dt_projs_bias.view(2, -1)[[0]].unsqueeze(0).contiguous() # (K, inner) -> (1, inner)
            dt_projs_bias_b1 = dt_projs_bias.view(2, -1)[[1]].unsqueeze(0).contiguous() # (K, inner) -> (1, inner)
            A_logs_b0 = A_logs.view(2, -1, N)[[0]].unsqueeze(0).view(-1, N).contiguous() # (K * D, N) -> (D, N)
            A_logs_b1 = A_logs.view(2, -1, N)[[1]].unsqueeze(0).view(-1, N).contiguous() # (K * D, N) -> (D, N)
            Ds_b0 = Ds.view(2, -1)[[0]].unsqueeze(0).contiguous().view(-1) # (K * D) -> (1 * D)
            Ds_b1 = Ds.view(2, -1)[[1]].unsqueeze(0).contiguous().view(-1) # (K * D) -> (1 * D)
            if no_einsum:
                x_dbl_b0 = F.conv1d(xs_b0.view(B, -1, L), x_proj_weight_b0.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=1)
                x_dbl_b1 = F.conv1d(xs_b1.view(B, -1, L_b1), x_proj_weight_b1.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=1)
                dts_b0, Bs_b0, Cs_b0 = torch.split(x_dbl_b0.view(B, 1, -1, L), [R, N, N], dim=2)
                dts_b1, Bs_b1, Cs_b1 = torch.split(x_dbl_b1.view(B, 1, -1, L_b1), [R, N, N], dim=2)
                dts_b0 = F.conv1d(dts_b0.contiguous().view(B, -1, L), dt_projs_weight_b0.view(D, -1, 1), groups=1)
                dts_b1 = F.conv1d(dts_b1.contiguous().view(B, -1, L_b1), dt_projs_weight_b1.view(D, -1, 1), groups=1)
            else:
                print('not support einsum, exit')
                exit()
            xs_b0 = xs_b0.view(B, -1, L).contiguous() # (b, k * d, l)
            dts_b0 = dts_b0.contiguous().view(B, -1, L)
            Bs_b0 = Bs_b0.contiguous().view(B, 1, N, L)
            Cs_b0 = Cs_b0.contiguous().view(B, 1, N, L)
            As_b0 = -torch.exp(A_logs_b0.to(torch.float))
            Ds_b0 = Ds_b0.to(torch.float)
            delta_bias_b0 = dt_projs_bias_b0.view(-1).to(torch.float)

            xs_b1 = xs_b1.view(B, -1, L_b1).contiguous() # (b, k * d, l)
            dts_b1 = dts_b1.contiguous().view(B, -1, L_b1)
            Bs_b1 = Bs_b1.contiguous().view(B, 1, N, L_b1)
            Cs_b1 = Cs_b1.contiguous().view(B, 1, N, L_b1)
            As_b1 = -torch.exp(A_logs_b1.to(torch.float))
            Ds_b1 = Ds_b1.to(torch.float)
            delta_bias_b1 = dt_projs_bias_b1.view(-1).to(torch.float)

            if force_fp32:
                xs_b0, dts_b0, Bs_b0, Cs_b0 = to_fp32(xs_b0, dts_b0, Bs_b0, Cs_b0)
                xs_b1, dts_b1, Bs_b1, Cs_b1 = to_fp32(xs_b1, dts_b1, Bs_b1, Cs_b1)
            ys_b0: torch.Tensor = selective_scan(
                xs_b0, dts_b0, As_b0, Bs_b0, Cs_b0, Ds_b0, delta_bias_b0, delta_softplus
            ).view(B, 1, -1, H, W).squeeze(1) # (B, C, H*W)
            ys_b1: torch.Tensor = selective_scan(
                xs_b1, dts_b1, As_b1, Bs_b1, Cs_b1, Ds_b1, delta_bias_b1, delta_softplus
            ).view(B, 1, -1, L_b1).squeeze(1) # (B, C, H*W//3)

            y_b1, y_b0 = ys_b1, ys_b0

            y_b0 = y_b0.view(B, -1, H, W)
            # y_hori_r = y_b1[:, :, :H_b1 * W_b1].flip(-1).view(B, -1, H_b1, W_b1)
            # y_vert = y_b1[:, :, H_b1 * W_b1:H_b1 * W_b1 * 2].view(B, -1, W_b1, H_b1).transpose(2, 3)
            # y_vert_r = y_b1[:,:, H_b1 * W_b1 * 2:].flip(-1).view(B, -1, W_b1, H_b1).transpose(2, 3)
            # y_b1 = y_hori_r + y_vert + y_vert_r
            y_b1,y_b0 = get_splited_xs(y_b1,y_b0, self.full_res_index, H_b1, W_b1)
            y_b1 = F.interpolate(y_b1, size=(H, W), mode='nearest')

            y = y_b0 + y_b1
        elif self.forward_type == 'axis':
            def axis_scan1d(
                    xs, proj_weight, proj_bias, dt_weight, dt_bias, _As, _Ds,
            ):
                _B, _K, _D, _L = xs.shape
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(x_proj_bias.view(-1) if proj_bias is not None else None), groups=_K)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, _K, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(_K * _D, -1, 1), groups=_K)
                else:
                    raise NotImplementedError

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, _K, N, _L)
                Cs = Cs.contiguous().view(_B, _K, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, _K, -1, _L)
                return ys


            axis_xs, axis_ys = axis_scan2d(x)
            As = -torch.exp(A_logs.to(torch.float)).view(K, -1, N)
            if H==W:
                axis_tensors = torch.cat([axis_xs, axis_ys], dim=1)
                axis_tensors = axis_scan1d(
                    axis_tensors,
                    proj_weight=x_proj_weight.view(K, -1, D)[:4].contiguous(),
                    proj_bias=(x_proj_bias.view(K, -1)[:4].contiguous() if x_proj_bias is not None else None),
                    dt_weight=dt_projs_weight.view(K, D, -1)[:4].contiguous(),
                    dt_bias=(dt_projs_bias.view(K, -1)[:4].contiguous() if dt_projs_bias is not None else None),
                    _As=As[:4].contiguous().view(-1, N),
                    _Ds=Ds.view(K, -1)[:4].contiguous().view(-1),
                )#B*H, 4, C, W+1
                axis_xs, axis_ys = axis_tensors.chunk(2, dim=1)
            else:
                axis_xs = axis_scan1d(
                    axis_xs,
                    proj_weight=x_proj_weight.view(K, -1, D)[:2].contiguous(),
                    proj_bias=(x_proj_bias.view(K, -1)[:2].contiguous() if x_proj_bias is not None else None),
                    dt_weight=dt_projs_weight.view(K, D, -1)[:2].contiguous(),
                    dt_bias=(dt_projs_bias.view(K, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                    _As=As[:2].contiguous().view(-1, N),
                    _Ds=Ds.view(K, -1)[:2].contiguous().view(-1),
                ) #B*H, 2, C, W+1
                axis_ys = axis_scan1d(
                    axis_ys,
                    proj_weight=x_proj_weight.view(K, -1, D)[2:4].contiguous(),
                    proj_bias=(x_proj_bias.view(K, -1)[2:4].contiguous() if x_proj_bias is not None else None),
                    dt_weight=dt_projs_weight.view(K, D, -1)[2:4].contiguous(),
                    dt_bias=(dt_projs_bias.view(K, -1)[2:4].contiguous() if dt_projs_bias is not None else None),
                    _As=As[2:4].contiguous().view(-1, N),
                    _Ds=Ds.view(K, -1)[2:4].contiguous().view(-1),
                )# B*W, 2, C, H+1
            xs, x_tokens, y_tokens = axis_merge2d(axis_xs, axis_ys, B, H, W)
            x_tokens = axis_scan1d(
                x_tokens,
                proj_weight=x_proj_weight.view(K, -1, D)[4:].contiguous(),
                proj_bias=(x_proj_bias.view(K, -1)[4:].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(K, D, -1)[4:].contiguous(),
                dt_bias=(dt_projs_bias.view(K, -1)[4:].contiguous() if dt_projs_bias is not None else None),
                _As=As[4:].contiguous().view(-1, N),
                _Ds=Ds.view(K, -1)[4:].contiguous().view(-1),
            )#(B, 2, C, H)
            y_tokens = axis_scan1d(
                y_tokens,
                proj_weight=x_proj_weight.view(K, -1, D)[4:].contiguous(),
                proj_bias=(x_proj_bias.view(K, -1)[4:].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(K, D, -1)[4:].contiguous(),
                dt_bias=(dt_projs_bias.view(K, -1)[4:].contiguous() if dt_projs_bias is not None else None),
                _As=As[4:].contiguous().view(-1, N),
                _Ds=Ds.view(K, -1)[4:].contiguous().view(-1),
            )#(B, 2, C, W)
            # Use out-of-place operations to avoid modifying views in-place
            x_tokens = torch.stack([x_tokens[:, 0], x_tokens[:, 1].flip(dims=[-1])], dim=1)
            y_tokens = torch.stack([y_tokens[:, 0], y_tokens[:, 1].flip(dims=[-1])], dim=1)
            x_tokens,y_tokens = x_tokens.sum(1),y_tokens.sum(1) # (B, C, H), (B, C, W)
            x_tokens = x_tokens.unsqueeze(-1).expand(-1, -1, -1, W) # (B, C, H, W)
            y_tokens = y_tokens.unsqueeze(-2).expand(-1, -1, H, -1) # (B, C, H, W)
            y = xs + x_tokens + y_tokens
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float) # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x_ori = x
            x = self.conv2d(x) # (b, d, h, w)
            x_b1 = None
        x = self.act(x)
        if self.forward_type == 'vms':
            x_b1 = self.act(self.conv2d_b1(x_ori))
            y = self.forward_core(x,x_b1 = x_b1)
        else:
            y = self.forward_core(x)

        if self.kwargs.get('add_se', False):
            if self.channel_first: y = self.se(y)
            else:
                y = y.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
                y = self.se(y)
                y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

# support: xv1a,xv2a,xv3a; 
# postfix: _cpos;_ocov;_ocov2;_ca,_ca1;_act;_mul;_onsigmoid,_onsoftmax,_ondwconv3,_onnone;


class SS2D(nn.Module, mamba_init, SS2Dv0, SS2Dv2, SS2Dv3):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
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
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        else:
            self.__initv2__(**kwargs)


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            if forward_type == 'mambav2':
                self.norm = norm_layer(hidden_dim)
                heads = kwargs.get('num_heads',[1,2,4,8])[kwargs.get('current_stage', 0)]
                self.op = Mamba2(
                    d_model=hidden_dim,
                    d_state=ssm_d_state,
                    d_conv=ssm_conv,
                    expand = int(ssm_ratio),
                    conv_bias=ssm_conv_bias,
                    headdim=hidden_dim*int(ssm_ratio) // heads,
                    chunk_size=256, #TODO: pass this as a parameter
                    #use_mem_eff_path = False,
                )
            elif forward_type == 'linearAttention':
                self.op = GatedLinearAttention(
                    hidden_dim,
                    kwargs.get('input_resolution', 14),
                    kwargs.get('num_heads',[1,2,4,8])[kwargs.get('current_stage', 0)],
                    drop_path=drop_path,
                    #norm_layer=norm_layer,
                )
            else:
                self.norm = norm_layer(hidden_dim)
                self.op = SS2D(
                    d_model=hidden_dim,
                    d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    dt_rank=ssm_dt_rank,
                    act_layer=ssm_act_layer,
                    # ==========================
                    d_conv=ssm_conv,
                    conv_bias=ssm_conv_bias,
                    # ==========================
                    dropout=ssm_drop_rate,
                    # bias=False,
                    # ==========================
                    initialize=ssm_init,
                    # ==========================
                    forward_type=forward_type,
                    channel_first=channel_first,
                    **kwargs,
                )
        
        self.drop_path = DropPath(drop_path)
        self.convFFN_branch = kwargs.get('convffn', False)
        self.channel_first = channel_first
        self.forward_type = forward_type
        if self.convFFN_branch:
            self.convFFN = ConvFFN(hidden_dim, expansion=kwargs.get('conv_ffn_ratio',2))
            self.mlp_branch = False
            self.convFFN_branch = True
            self.norm2 = norm_layer(hidden_dim)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.forward_type == 'linearAttention':
                x = self.op(x)
            elif self.forward_type == 'mambav2':
                x = x + self.drop_path(self.op(self.norm(x)))
            else:
                if self.post_norm:
                    x = x + self.drop_path(self.norm(self.op(x)))
                else:
                    x = x + self.drop_path(self.op(self.norm(x)))
        if self.convFFN_branch:
            if self.channel_first:
                x = x + self.drop_path(self.norm2(self.convFFN(x)))
            else:
                x = x + self.drop_path(
                    self.convFFN(self.norm2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous())
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


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
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        # =========================
        posembed=False,
        imgsize=224,
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
            linearattn = self._make_patch_embed_linearattn,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3,
            linearattn = self._make_downsample_linearattn,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()

        if type(forward_type) is str: forward_type = [forward_type] * self.num_layers

        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()
            kwargs['current_stage'] = i_layer
            kwargs['input_resolution'] = [imgsize // (patch_size * 2 ** i_layer)]*2 #FIXME: pass this as a parameter
            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type[i_layer],
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                #==================
                **kwargs,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_linearattn(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None),
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_linearattn(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        ratio = 4
        return nn.Sequential(
            nn.Conv2d(dim, int(ratio*out_dim), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(out_dim * ratio), int(out_dim * ratio), kernel_size=3, stride=2, padding=1,
                      groups=int(out_dim * ratio)),
            nn.ReLU(),
            nn.Conv2d(int(out_dim * ratio), out_dim, kernel_size=1),
            nn.BatchNorm2d(int(out_dim)),
        )
    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                **kwargs,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanOflex": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanCore": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanNRow": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            #"prim::PythonOp.MambaChunkScanCombinedFn": partial(mamba_chunk_scan_combined_flop_jit, flops_fn=mamba_chunk_scan_combined_fn, verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        try:
            Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        except Exception as e:
            print('get exception', e)
            print('Error in flop_count, set to default value 1e6')
            return 1e6
        del model, input
        return sum(Gflops.values()) * 1e9
        #return f"params {params} GFLOPs {sum(Gflops.values())}"

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

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

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
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model_ema"):
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
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        
        return outs

