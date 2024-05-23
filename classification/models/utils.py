import torch
import torch.nn as nn
import math
from functools import partial
from einops import rearrange, repeat
import torch.nn.functional as F

try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class ConvFFN(nn.Module):

    def __init__(self, channels, expansion=2, drop=0.2):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0)
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class SEModule(nn.Module):
    def __init__(self, channels, reduction=4, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            act_layer(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() if gate_layer == 'sigmoid' else nn.HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# cross selective scan ===============================

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # (b, k, c, l)
        return xs  # (b, k, c, l)

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


class ThreeScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # (b, k, c, l)
        return xs[:, 1:]  # (b, k, c, l)

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys[:, 1:3] = ys[:, 1:3].flip(dims=[-1])
        ys[:, 0] = ys[:, 0] + ys[:, 2]
        y = ys[:, 0].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L) + ys[:, 1]
        return y.view(B, -1, H, W)


class ThreeMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys[:, 1:3] = ys[:, 1:3].flip(dims=[-1])
        ys[:, 0] = ys[:, 0] + ys[:, 2]
        y = ys[:, 0].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1) + ys[:, 1]

        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)[:,1:]
        return xs, None, None

class SelectiveScan(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

# class CrossGroupScan(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x: torch.Tensor):
#         B, C, H, W = x.shape
#         ctx.shape = (B, C, H, W)
#         xs = x.new_empty((B, 4, C, H * W))
#         xs[:, 0] = x.flatten(2, 3)
#         xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
#         xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # (b, k, c, l)
#         return xs
#
#     @staticmethod
#     def backward(ctx, ys: torch.Tensor):
#         # out: (b, k, d, l)
#         B, C, H, W = ctx.shape
#         y = ys.new_empty((B, 4, C, H*W))
#         y[:, 0] = ys[:, 0]
#         y[:, 1] = ys[:, 1].view(B,C,H,W).transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)
#         y[:, 2] = ys[:, 2].flip(dims=[-1])
#         y[:, 3] = ys[:, 3].flip(dims=[-1]).view(B, C, H, W).transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)
#         y = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3]
#         return y.view(B, -1, H, W)
#
#
# class CrossGroupMerge(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, ys: torch.Tensor):
#         B, K, C, H, W = ys.shape
#         ctx.shape = (H, W)
#         y = ys.new_empty((B, 4, C, H, W))
#         y[:, 0] = ys[:, 0]
#         y[:, 1] = ys[:, 1].transpose(dim0=2, dim1=3).contiguous()
#         y[:, 2] = ys[:, 2].flatten(2,3).flip(dims=[-1]).view(B, C, H, W)
#         y[:, 3] = ys[:, 3].flatten(2,3).flip(dims=[-1]).view(B, C, H, W).transpose(dim0=2, dim1=3).contiguous()
#         return y.view(B, -1, H, W).view(B, 4*C, H*W)
#
#     @staticmethod
#     def backward(ctx, x: torch.Tensor):
#         # B, D, L = x.shape
#         # out: (b, k, d, l)
#         H, W = ctx.shape
#         B, C, L = x.shape
#         xs = x.new_empty((B, 4, C // 4, L))
#         xs[:, 0] = x[:, 0:C//4]
#         xs[:, 1] = x[:,C//4:C//2].view(B, C//4, H, W).transpose(dim0=2, dim1=3).flatten(2, 3).contiguous()
#         xs[:, 2] = x[:,C//2:C//4*3].flip(dims=[-1])
#         xs[:, 3] = x[:,C//4*3:C].flip(dims=[-1]).view(B, C//4, H, W).transpose(dim0=2, dim1=3).flatten(2, 3).contiguous()
#         xs = xs.view(B, 4, C//4, H, W)
#         return xs, None, None

class CrossScan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 2, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        return xs  # (b, k, c, l)

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs = xs.view(B, 2, C, H, W)
        return xs, None, None

class CrossScan3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 3, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2] = torch.flip(xs[:, 0], dims=[-1])  # (b, k, c, l)
        return xs[:,:3]  # (b, k, c, l)

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L) \
            + ys[:, 2].flip(dims=[-1])
        return y.view(B, -1, H, W)

class CrossMerge3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1) \
            + ys[:, 2].flip(dims=[-1])
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 3, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2] = torch.flip(xs[:, 0], dims=[-1])
        xs = xs.view(B, 3, C, H, W)
        return xs, None, None

class SSMInterBlock(nn.Module):
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
        recurrent = False,
        recurrent_stride = 0,
        # ======================
        **kwargs,):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
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

        self.forward_core = self.forward_corev2
        self.K = 1
        self.K2 = self.K

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()


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
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # recurrent =======================================
        self.recurrent = recurrent
        self.recurrent_stride = recurrent_stride
        # other kwargs =======================================
        self.kwargs = kwargs
        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
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


    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 2, 1).contiguous() # (b, d, l)
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = selective_scan_flatten(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
            recurrent=self.recurrent, recurrent_stride=self.recurrent_stride,
            **self.kwargs,
        ) # (B, L, d)
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        # x: (b, h, w, d, blocks)
        b, h, w, d, blocks = x.shape
        x = x.permute(0,1,2,4,3).flatten(0,2).contiguous() # (b*h*w, blocks, d)
        xz = self.in_proj(x)

        if self.disable_z_act:
            x, z = xz.chunk(2, dim=-1)  # (b*h*w,blocks, d)
            x = self.act(x)
        else:
            xz = self.act(xz)
            x, z = xz.chunk(2, dim=-1)  # (b*h*w,blocks, d)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y)) # (b*h*w, blocks, d)
        out = out.view(b, h, w, blocks, d).permute(0,1,2,4,3).contiguous() # (b, h, w, d, blocks)
        out = out[:, :, :, :, -1] # (b, h, w, d)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SSMultiScale(nn.Module):
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
            d_conv=3,  # < 2 means no conv
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
            recurrent=False,
            recurrent_stride=0,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.d_inner = d_inner
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        b1_ratio = kwargs.get('b1_ratio', 0.5)
        b0_dim, b1_dim = int(d_inner*(1-b1_ratio)), int(d_inner*b1_ratio)
        self.b0_dim = b0_dim
        self.b1_dim = b1_dim

        # disable z act ======================================
        self.disable_z_act = False
        self.out_norm_b0 = nn.LayerNorm(b0_dim)
        self.out_norm_b1 = nn.LayerNorm(b1_dim)
        self.K_b0 = 4  # branch 0, full resolution
        if kwargs.get('b1_seq', False):
            self.K_b1 = 1
        else:
            self.K_b1 = 4  # branch 1, downsampled resolution


        self.forward_core = self.forward_core_multiscale
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        self.conv2d_b0 = nn.Conv2d(
            in_channels=b0_dim,
            out_channels=b0_dim,
            groups=b0_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )  # branch 0, convert B, C, H, W to B, C, H, W

        b1_stride = 2
        self.conv2d_b1 = nn.Conv2d(
            in_channels=b1_dim,
            out_channels=b1_dim,
            groups=b1_dim,
            bias=conv_bias,
            kernel_size=7,
            stride=b1_stride,
            padding=3,
            **factory_kwargs,
        )  # bracnh 1, convert B, C, H, W to B, C, H//2, W//2

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj_b0 = [
            nn.Linear(b0_dim, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K_b0)
        ]
        self.x_proj_weight_b0 = nn.Parameter(torch.stack([t.weight for t in self.x_proj_b0], dim=0))  # (K, N, inner)
        del self.x_proj_b0

        self.x_proj_b1 = [
            nn.Linear(b1_dim, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K_b1)
        ]
        self.x_proj_weight_b1 = nn.Parameter(torch.stack([t.weight for t in self.x_proj_b1], dim=0))  # (K, N, inner)
        del self.x_proj_b1

        # dt proj ============================
        self.dt_projs_b0 = [
            self.dt_init(self.dt_rank, b0_dim, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K_b0)
        ]
        self.dt_projs_weight_b0 = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs_b0], dim=0))  # (K, inner, rank)
        self.dt_projs_bias_b0 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_b0], dim=0))  # (K, inner)
        del self.dt_projs_b0

        self.dt_projs_b1 = [
            self.dt_init(self.dt_rank, b1_dim, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K_b1)
        ]
        self.dt_projs_weight_b1 = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs_b1], dim=0))  # (K, inner, rank)
        self.dt_projs_bias_b1 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_b1], dim=0))
        del self.dt_projs_b1

        # A, D =======================================
        self.A_logs_b0 = self.A_log_init(self.d_state, b0_dim, copies=self.K_b0, merge=True)  # (K * D, N)
        self.Ds_b0 = self.D_init(b0_dim, copies=self.K_b0, merge=True)  # (K * D)

        self.A_logs_b1 = self.A_log_init(self.d_state, b1_dim, copies=self.K_b1, merge=True)  # (K * D, N)
        self.Ds_b1 = self.D_init(b1_dim, copies=self.K_b1, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.kwargs = kwargs

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
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

    def forward_core_multiscale(self, xs: list, nrows=-1, channel_first=False):
        nrows = 1
        x_b0, x_b1 = xs
        x_b0 = [x.permute(0, 2, 1).contiguous() for x in x_b0]
        if self.kwargs.get('b1_seq', False):
            x_b1 = x_b1.permute(0, 2, 1).contiguous()
        else:
            x_b1 = [x.permute(0, 2, 1).contiguous() for x in x_b1]

        # if self.ssm_low_rank:
        #     xs = [self.in_rank(x) for x in xs]

        for i in range(self.K_b0):
            x_b0[i] = selective_scan_flatten(
                x_b0[i], self.x_proj_weight_b0[[i]], None, self.dt_projs_weight_b0[[i]], self.dt_projs_bias_b0[[i]],
                self.A_logs_b0[i * self.b0_dim:(i + 1) * self.b0_dim],
                self.Ds_b0[i * self.b0_dim:(i + 1) * self.b0_dim],
                getattr(self, "out_norm_b0", None),
                nrows=nrows, delta_softplus=True, force_fp32=self.training,
                **self.kwargs,
            )
        if self.kwargs.get('b1_seq',False):
            x_b1 = selective_scan_flatten(
                x_b1, self.x_proj_weight_b1, None, self.dt_projs_weight_b1, self.dt_projs_bias_b1,
                self.A_logs_b1, self.Ds_b1,
                getattr(self, "out_norm_b1", None),
                nrows=nrows, delta_softplus=True, force_fp32=self.training,
                **self.kwargs,
            )
        else:
            for i in range(self.K_b1):
                x_b1[i] = selective_scan_flatten(
                    x_b1[i], self.x_proj_weight_b1[[i]], None, self.dt_projs_weight_b1[[i]], self.dt_projs_bias_b1[[i]],
                    self.A_logs_b1[i * self.b1_dim:(i + 1) * self.b1_dim],
                    self.Ds_b1[i * self.b1_dim:(i + 1) * self.b1_dim],
                    getattr(self, "out_norm_b1", None),
                    nrows=nrows, delta_softplus=True, force_fp32=self.training,
                    **self.kwargs,
                )
        ys = [x_b0, x_b1]
        # if self.ssm_low_rank:
        #     ys = [self.out_rank(x) for x in ys]
        return ys

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)

        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        b, h, w, d = x.shape
        if not self.disable_z_act:
            z = self.act(z)  # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()  # (b, d, h, w)
        b1_ratio = self.kwargs.get('b1_ratio', 0.5)
        x_b0, x_b1 = x[:, :int(d * (1 - b1_ratio))], x[:, int(d * (1 - b1_ratio)):]
        x_b0 = self.act(self.conv2d_b0(x_b0))  # (b, d//2, h, w)
        x_b1 = self.act(self.conv2d_b1(x_b1))
        h_b1, w_b1 = x_b1.shape[2:]
        # horizontal scan
        x_hori_b0 = x_b0.flatten(2)
        x_hori_b1 = x_b1.flatten(2)
        # reverse horizontal scan
        x_horir_b0 = x_b0.flatten(2).flip(-1)
        x_horir_b1 = x_b1.flatten(2).flip(-1)
        # vertical scan
        x_vert_b0 = x_b0.transpose(2, 3).flatten(2)
        x_vert_b1 = x_b1.transpose(2, 3).flatten(2)
        # reverse vertical scan
        x_vertr_b0 = x_b0.transpose(2, 3).flatten(2).flip(-1)
        x_vertr_b1 = x_b1.transpose(2, 3).flatten(2).flip(-1)

        x_b0 = [x_hori_b0,x_horir_b0, x_vert_b0, x_vertr_b0]
        x_b1 = [x_hori_b1,x_horir_b1, x_vert_b1, x_vertr_b1]
        if self.kwargs.get('b1_seq',False):
            x_b1 = torch.cat(x_b1, dim=-1)
        x = [x_b0, x_b1]

        y = self.forward_core(x, channel_first=(self.d_conv > 1))

        y_b0, y_b1 = y

        #reverse scan
        y_hori_b0 = y_b0[0].view(b, h, w, -1)
        y_horir_b0 = y_b0[1].flip(1).view(b, h, w, -1)
        y_vert_b0 = y_b0[2].view(b, w, h, -1).transpose(1, 2)
        y_vertr_b0 = y_b0[3].flip(1).view(b, w, h, -1).transpose(1, 2)
        y_b0 = y_hori_b0 + y_vert_b0 + y_horir_b0 + y_vertr_b0
        y_b0 = y_b0.contiguous()
        if self.kwargs.get('b1_seq', False):
            y_hori_b1 = y_b1[:, :h_b1 * w_b1].view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
            y_horir_b1 = y_b1[:, h_b1 * w_b1:h_b1 * w_b1 * 2].flip(-2).view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
            y_vert_b1 = y_b1[:, h_b1 * w_b1 * 2:h_b1 * w_b1 * 3].view(b, h_b1, w_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
            y_vertr_b1 = y_b1[:, h_b1 * w_b1 * 3:].flip(-2).view(b, h_b1, w_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
        else:
            y_hori_b1 = y_b1[0].view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
            y_horir_b1 = y_b1[1].flip(1).view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
            y_vert_b1 = y_b1[2].view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
            y_vertr_b1 = y_b1[3].flip(1).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
        y_b1 = y_hori_b1 + y_vert_b1 + y_horir_b1 + y_vertr_b1
        y_b1 = F.interpolate(y_b1, size=(h, w), mode='bilinear', align_corners=False)
        y_b1 = y_b1.permute(0, 2, 3, 1).contiguous()

        y = torch.cat([y_b0, y_b1], dim=-1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

def selective_scan_flatten( x: torch.Tensor=None,
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
    recurrent = False,
    recurrent_stride = 0,
    merge = False,
    **kwargs,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, L, D = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape

    xs = x.transpose(dim0=1, dim1=2).unsqueeze(1).contiguous()

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
    ) # (B, K, C, L)
    ys = ys.view(B, K, -1, L)
    y = ys.squeeze(1).transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
    y = out_norm(y) # (B, L, C)

    return (y.to(x.dtype) if to_dtype else y)


def x_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        force_fp32=True,
        **kwargs,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
    K, D, R = dt_projs_weight.shape
    if K == 1:
        B, D, L = x.shape
    else:
        B, D, H, W = x.shape
        L = H * W
    D, N = A_logs.shape


    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    if K==1:
        xs = x.unsqueeze(1).contiguous()
    elif K==2:
        xs = CrossScan2.apply(x)
    elif K==3:
        xs = CrossScan3.apply(x)
    elif K==4:
        xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
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


    if K==1:
        y = ys.view(B, K, -1, L).squeeze(1) # (B, C, L)
    else:
        ys = ys.view(B, K, -1, H, W)
    if K==2:
        y: torch.Tensor = CrossMerge2.apply(ys)
    elif K==3:
        y: torch.Tensor = CrossMerge3.apply(ys)
    elif K==4:
        y: torch.Tensor = CrossMerge.apply(ys)

    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = out_norm(y)
    if K!=1: y = y.view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y), {'ys':ys, 'xs':xs, 'dts':dts, 'As':A_logs, 'Bs':Bs, 'Cs':Cs, 'Ds':Ds, 'delta_bias':delta_bias}

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)




