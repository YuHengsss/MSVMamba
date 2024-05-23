import os
from functools import partial
from typing import Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from PIL import Image
import math
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from timm.utils import accuracy, AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from fvcore.nn import FlopCountAnalysis, flop_count
import random

import matplotlib.pyplot as plt
def check_path(path: str):
    if not os.path.exists(path): os.makedirs(path)
def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

# base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# models_path = os.path.join(base_path, 'classification', 'models')
# sys.path.insert(0, models_path)
# import vmamba
sys.path.append('../classification/models')
vmamba = import_abspy(
    "vmamba",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/models"),
)
VSSM: nn.Module = vmamba.VSSM
SS2D: nn.Module = vmamba.SS2D
VSSBlock: nn.Module = vmamba.VSSBlock
Mlp: nn.Module = vmamba.Mlp
#gMlp: nn.Module = vmamba.gMlp
DropPath: nn.Module = vmamba.DropPath
# SelectiveScanOflex: nn.Module = vmamba.SelectiveScanOflex
# CrossScanTriton: nn.Module = vmamba.CrossScanTriton
# CrossMergeTriton: nn.Module = vmamba.CrossMergeTriton
# CrossScanTriton1b1: nn.Module = vmamba.CrossScanTriton1b1

this_path = os.path.dirname(os.path.abspath(__file__))

from erf import visualize

visualize_attnmap = visualize.visualize_attnmap
visualize_attnmaps = visualize.visualize_attnmaps


def get_dataloader(batch_size=64, root="./val", img_size=224, sequential=True):
    from torch.utils.data import SequentialSampler, DistributedSampler, DataLoader
    size = int((256 / 224) * img_size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    dataset = datasets.ImageFolder(root, transform=transform)
    if sequential:
        sampler = SequentialSampler(dataset)
    else:
        sampler = DistributedSampler(dataset)

    data_loader = DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    return data_loader


def denormalize(image: torch.Tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    if len(image.shape) == 2:
        image = (image.cpu() * 255).to(torch.uint8).numpy()
    elif len(image.shape) == 3:
        C, H, W = image.shape
        image = image.cpu() * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    image = Image.fromarray(image)
    return image

def vis_channel_decay(ah_f_sorted, aw_f_sorted, ah_b_sorted, aw_b_sorted, dst_path):
    # draw four plot for ah_f, aw_f, ah_b, aw_b, x-axis is range(D), y-axis is the value of ah_f_sorted, aw_f_sorted, ah_b_sorted, aw_b_sorted

    fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=100)

    # Plot for ah_f_sorted
    axs[0, 0].plot(ah_f_sorted.detach().numpy())
    axs[0, 0].set_title('Forward Decay Along Height')
    axs[0, 0].set_xlabel('Channel Index')
    axs[0, 0].set_ylabel('Mean Activation')

    # Plot for aw_f_sorted
    axs[0, 1].plot(aw_f_sorted.detach().numpy())
    axs[0, 1].set_title('Forward Decay Along Width')
    axs[0, 1].set_xlabel('Channel Index')
    axs[0, 1].set_ylabel('Mean Activation')

    # Plot for ah_b_sorted
    axs[1, 0].plot(ah_b_sorted.detach().numpy())
    axs[1, 0].set_title('Backward Decay Along Height')
    axs[1, 0].set_xlabel('Channel Index')
    axs[1, 0].set_ylabel('Mean Activation')

    # Plot for aw_b_sorted
    axs[1, 1].plot(aw_b_sorted.detach().numpy())
    axs[1, 1].set_title('Backward Decay Along Width')
    axs[1, 1].set_xlabel('Channel Index')
    axs[1, 1].set_ylabel('Mean Activation')

    #adjust label, title size and x, y axis size
    for ax in axs.flat:
        ax.label_outer()
        ax.title.set_size(20)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.tick_params(axis='both', which='major', labelsize=15)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(dst_path)
    plt.close(fig)


@torch.no_grad()
def attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, with_ws=True, with_dt=False, only_ws=False, ret="all",
                  ratio=1, verbose=False, half=False):
    printlog = print if verbose else lambda *args, **kwargs: None
    printlog(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)

    B, G, N, L = Bs.shape
    GD, N = As.shape
    D = GD // G
    H, W = int(math.sqrt(L)), int(math.sqrt(L))
    mask = torch.tril(dts.new_ones((L, L)))
    dts = torch.nn.functional.softplus(dts + delta_bias[:, None]).view(B, G, D, L)
    dw_logs = As.view(G, D, N)[None, :, :, :,None] * dts[:, :, :, None, :]  # (B, G, D, N, L)
    ws = torch.cumsum(dw_logs, dim=-1).exp()
    # ave ws in first group, D,N dim
    ave_ws = torch.cumsum(dw_logs.flip(-1), dim=-1).exp() # (B, G, D, N, L)


    if only_ws:
        Qs, Ks = ws, 1 / ws.clamp(min=1e-20)
    else:
        Qs, Ks = Cs[:, :, None, :, :], Bs[:, :, None, :, :] # (B, G, 1, N, L), (B, G, 1, N, L)
        if with_ws:
            Qs, Ks = Qs * ws, Ks / ws.clamp(min=1e-20)
    if with_dt:
        Ks = Ks * dts.view(B, G, D, 1, L) # (B, G, D, N, L)

    printlog(ws.shape, Qs.shape, Ks.shape)
    printlog("Bs", Bs.max(), Bs.min(), Bs.abs().min())
    printlog("Cs", Cs.max(), Cs.min(), Cs.abs().min())
    printlog("ws", ws.max(), ws.min(), ws.abs().min())
    printlog("Qs", Qs.max(), Qs.min(), Qs.abs().min())
    printlog("Ks", Ks.max(), Ks.min(), Ks.abs().min())
    _Qs, _Ks = Qs.view(-1, N, L), Ks.view(-1, N, L) # (B*G, N, L), (B*G, N, L)
    attns = (_Qs.transpose(1, 2) @ _Ks).view(B, G, -1, L, L)
    attns = attns.mean(dim=2) * mask # (B, G, L, L)

    attn0 = attns[:, 0, :].view(B, -1, L, L)
    attn1 = attns[:, 1, :].view(-1, H, W, H, W).permute(0, 2, 1, 4, 3).contiguous().view(B, -1, L, L)
    attn2 = attns[:, 2, :].view(-1, L, L).flip(dims=[-2]).flip(dims=[-1]).contiguous().view(B, -1, L, L)
    attn3 = attns[:, 3, :].view(-1, L, L).flip(dims=[-2]).flip(dims=[-1]).contiguous().view(B, -1, L, L)
    attn3 = attn3.view(-1, H, W, H, W).permute(0, 2, 1, 4, 3).contiguous().view(B, -1, L, L)

    if ret in ["ao0"]:
        attn = attns[:, 0, :].view(B, -1, L, L).mean(dim=1)
    elif ret in ["ao1"]:
        attn = attns[:, 1, :].view(B, -1, L, L).mean(dim=1)
    elif ret in ["ao2"]:
        attn = attns[:, 2, :].view(B, -1, L, L).mean(dim=1)
    elif ret in ["ao3"]:
        attn = attns[:, 3, :].view(B, -1, L, L).mean(dim=1)
    elif ret in ["a0"]:
        attn = attn0.mean(dim=1)
    elif ret in ["a1"]:
        attn = attn1.mean(dim=1)
    elif ret in ["a2"]:
        attn = attn2.mean(dim=1)
    elif ret in ["a3"]:
        attn = attn3.mean(dim=1)
    elif ret in ["a0a2"]:
        attn = (attn0 + attn2).mean(dim=1)
    elif ret in ["a1a3"]:
        attn = (attn1 + attn3).mean(dim=1)
    elif ret in ["a0a1"]:
        attn = (attn0 + attn1).mean(dim=1)
    elif ret in ["a1a2a3"]:
        attn = (attn1 + attn2 + attn3).mean(dim=1)
    elif ret in ["all"]:
        attn = (attn0 + attn1 + attn2 + attn3).mean(dim=1) # (B, L, L)
    elif ret in ['ave_ws_a0']:
        ave_ws = ave_ws[bidx, 0, :, :].sum(-2).sum(-2) / (N * D)  # L
        ave_ws = ave_ws.flip(-1).view(int(math.sqrt(L)), int(math.sqrt(L)))
        return ave_ws
    elif ret in ['ave_ws_a1']:
        if half:
            h_a1, w_a1 = int(math.sqrt(L) // 2), int(math.sqrt(L) // 2)
            dw_logs_a1 = dw_logs[:, 1].view(B, D*N, int(math.sqrt(L)), int(math.sqrt(L)))
            dw_logs_a1 = F.interpolate(dw_logs_a1, size=(h_a1, w_a1), mode='bilinear', align_corners=False)
            dw_logs_a1 = dw_logs_a1.view(B, D, N, h_a1*w_a1)
            ave_ws = torch.cumsum(dw_logs_a1.flip(-1), dim=-1).exp()  # (B, D, N, L//4)
            L = h_a1 * w_a1
            ave_ws = ave_ws[bidx, :, :].sum(-2).sum(-2) / (N * D)  # L
        else:
            ave_ws = ave_ws[bidx, 1, :, :].sum(-2).sum(-2) / (N * D)  # L
        ave_ws = ave_ws.flip(-1).view(int(math.sqrt(L)), int(math.sqrt(L)))
        return ave_ws.transpose(0, 1)
    elif ret in ['ave_ws_a2']:
        if half:
            h_a1, w_a1 = int(math.sqrt(L) // 2), int(math.sqrt(L) // 2)
            dw_logs_a1 = dw_logs[:, 2].view(B, D * N, int(math.sqrt(L)), int(math.sqrt(L)))
            dw_logs_a1 = F.interpolate(dw_logs_a1, size=(h_a1, w_a1), mode='bilinear', align_corners=False)
            dw_logs_a1 = dw_logs_a1.view(B, D, N, h_a1 * w_a1)
            ave_ws = torch.cumsum(dw_logs_a1.flip(-1), dim=-1).exp()  # (B, D, N, L//4)
            L = h_a1 * w_a1
            ave_ws = ave_ws[bidx, :, :].sum(-2).sum(-2) / (N * D)  # L
        else:
            ave_ws = ave_ws[bidx, 2, :, :].sum(-2).sum(-2) / (N * D)  # L
        ave_ws = ave_ws.view(int(math.sqrt(L)), int(math.sqrt(L)))
        return ave_ws
    elif ret in ['ave_ws_a3']:
        if half:
            h_a1, w_a1 = int(math.sqrt(L) // 2), int(math.sqrt(L) // 2)
            dw_logs_a1 = dw_logs[:, 3].view(B, D * N, int(math.sqrt(L)), int(math.sqrt(L)))
            dw_logs_a1 = F.interpolate(dw_logs_a1, size=(h_a1, w_a1), mode='bilinear', align_corners=False)
            dw_logs_a1 = dw_logs_a1.view(B, D, N, h_a1 * w_a1)
            ave_ws = torch.cumsum(dw_logs_a1.flip(-1), dim=-1).exp()  # (B, D, N, L//4)
            L = h_a1 * w_a1
            ave_ws = ave_ws[bidx, :, :].sum(-2).sum(-2) / (N * D)  # L
        else:
            ave_ws = ave_ws[bidx, 3, :, :].sum(-2).sum(-2) / (N * D)  # L
        ave_ws = ave_ws.view(int(math.sqrt(L)), int(math.sqrt(L)))
        return ave_ws.transpose(0, 1)

    return ratio * attn[bidx, :, :]


def get_centra_attn_ratio(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, with_ws=True, with_dt=False, only_ws=False, ret="all",
                  ratio=1, verbose=False, half=False):
    printlog = print if verbose else lambda *args, **kwargs: None
    printlog(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)

    B, G, N, L = Bs.shape
    GD, N = As.shape
    D = GD // G
    H, W = int(math.sqrt(L)), int(math.sqrt(L))
    mask = torch.tril(dts.new_ones((L, L)))
    dts = torch.nn.functional.softplus(dts + delta_bias[:, None]).view(B, G, D, L)
    dw_logs = As.view(G, D, N)[None, :, :, :,None] * dts[:, :, :, None, :]  # (B, G, D, N, L)
    ws = torch.cumsum(dw_logs, dim=-1).exp()
    # ave ws in first group, D,N dim
    H, W = int(math.sqrt(L)), int(math.sqrt(L))
    ave_ws = dw_logs#torch.cumsum(dw_logs.flip(-1), dim=-1).exp() # (B, G, D, N, L)
    ave_ws_a0 = ave_ws[bidx, 0, :, :].view(D, N, H, W) #(D, N, H, W)
    ave_ws_a1 = ave_ws[bidx, 1, :, :].view(D, N, W, H).transpose(-1, -2) #(D, N, H, W)
    ave_ws_a2 = ave_ws[bidx, 2, :, :].flip(-1).view(D, N, H, W) #(D, N, H, W)
    ave_ws_a3 = ave_ws[bidx, 3, :, :].flip(-1).view(D, N, W, H).transpose(-1, -2) #(D, N, H, W)
    ws_a0_center = ave_ws_a0.flatten(-2,-1)[:,:,:L//2-W//2]
    ws_a1_center = ave_ws_a1.flatten(-2,-1)[:,:,:L//2-H//2-1]
    ws_a2_center = ave_ws_a2.flatten(-2,-1)[:,:,L//2-W//2:]
    ws_a3_center = ave_ws_a3.flatten(-2,-1)[:,:,L//2-H//2-1:]

    ws_a0_center = torch.cumsum(ws_a0_center.flip(-1), dim=-1).exp().flip(-1)
    ws_a1_center = torch.cumsum(ws_a1_center.flip(-1), dim=-1).exp().flip(-1)
    ws_a2_center = torch.cumsum(ws_a2_center, dim=-1).exp()
    ws_a3_center = torch.cumsum(ws_a3_center, dim=-1).exp()

    ws_ah = torch.cat([ws_a0_center, ws_a2_center], dim=-1).view(D, N, H, W) # (D, N, H, W)
    ws_aw = torch.cat([ws_a1_center, ws_a3_center], dim=-1).view(D, N, H, W).transpose(-1, -2) # (D, N, H, W)
    ave_ah = ws_ah.mean(1)
    ave_aw = ws_aw.mean(1)
    #ws_ratio = ws_ah.clip(1e-9,1e9) / ws_aw.clip(1e-9,1e9)
    #ave_ws_ratio = ws_ratio.clip(1e-9,1e9).mean(dim=0).mean(dim=0) # (H, W)
    ave_ratio = torch.max(ave_ah / ave_aw, ave_aw / ave_ah)
    # visualize_attnmap(ave_ah,dpi=50)
    # visualize_attnmap(ave_aw, dpi=50)
    # visualize_attnmap(ave_ratio, dpi=50)

    return ave_ah, ave_aw


def add_hook(model: nn.Module):
    ss2ds = []
    for layer in model.layers:
        _ss2ds = []
        for blk in layer.blocks:
            ss2d = blk.op
            setattr(ss2d, "__DEBUG__", True)
            _ss2ds.append(ss2d)
        ss2ds.append(_ss2ds)
    return model, ss2ds


def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith("backbone."):
            new_state_dict[k[len("backbone."):]] = state_dict[k]
    return new_state_dict


def visual_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, with_ws=False, with_dt=False, only_ws=False, ratio=1, tag="bcs",
                 H=56, W=56, front_point=(0.5, 0.5), front_back=(0.7, 0.8), showpath=os.path.join(this_path, "show")
                 , half=False):
    kwargs = dict(with_ws=with_ws, with_dt=with_dt, only_ws=only_ws, ratio=ratio)
    visualize_attnmap(
        attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ave_ws_a0", half=half, **kwargs),
        savefig=f"{showpath}/{tag}_ave_ws_a0.jpg"
    )
    visualize_attnmap(
        attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ave_ws_a1", half=half,  **kwargs),
        savefig=f"{showpath}/{tag}_ave_ws_a1.jpg",
    )
    visualize_attnmap(
        attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ave_ws_a2", half=half, **kwargs),
        savefig=f"{showpath}/{tag}_ave_ws_a2.jpg",
    )
    visualize_attnmap(
        attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ave_ws_a3", half=half, **kwargs),
        savefig=f"{showpath}/{tag}_ave_ws_a3.jpg"
    )

    visualize_attnmap(
        attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs),
        savefig=f"{showpath}/{tag}_merge.jpg"
    )
    visualize_attnmap(
        torch.diag(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)).view(H, W),
        savefig=f"{showpath}/{tag}_attn_diag.jpg")  # self attention
    visualize_attnmap(
        torch.diag(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs)).view(H, W),
        savefig=f"{showpath}/{tag}_attn_diag_a0.jpg")
    visualize_attnmap(
        torch.diag(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs)).view(H, W),
        savefig=f"{showpath}/{tag}_attn_diag_a1.jpg")
    visualize_attnmap(
        torch.diag(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs)).view(H, W),
        savefig=f"{showpath}/{tag}_attn_diag_a2.jpg")
    visualize_attnmap(
        torch.diag(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs)).view(H, W),
        savefig=f"{showpath}/{tag}_attn_diag_a3.jpg")
    visualize_attnmap(
        torch.diag(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1a2a3", **kwargs)).view(H, W),
        savefig=f"{showpath}/{tag}_attn_diag_a1a2a3.jpg")
    visualize_attnmap(
        torch.mean(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs), dim=-1).view(H, W),
        savefig=f"{showpath}/{tag}_attn_sum.jpg")  # self attention
    visualize_attnmap(
        torch.mean(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs), dim=-1).view(H, W),
        savefig=f"{showpath}/{tag}_attn_sum_a0.jpg")
    visualize_attnmap(
        torch.mean(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs), dim=-1).view(H, W),
        savefig=f"{showpath}/{tag}_attn_sum_a1.jpg")
    visualize_attnmap(
        torch.mean(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs), dim=-1).view(H, W),
        savefig=f"{showpath}/{tag}_attn_sum_a2.jpg")
    visualize_attnmap(
        torch.mean(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs), dim=-1).view(H, W),
        savefig=f"{showpath}/{tag}_attn_sum_a3.jpg")
    visualize_attnmap(
        torch.mean(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1a2a3", **kwargs), dim=-1).view(H, W),
        savefig=f"{showpath}/{tag}_attn_sum_a1a2a3.jpg")
    # visualize_attnmap(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[int(front_point[0] * H * W + front_point[1] * W)].view(H, W), savefig=f"{showpath}/{tag}_attn_front.jpg") # front attention
    # visualize_attnmap(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[int(front_back[0] * H * W + front_back[1] * W)].view(H, W), savefig=f"{showpath}/{tag}_attn_back.jpg") # back attention
    visualize_attnmaps([
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao0", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao1", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao2", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao3", **kwargs), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs), ""),
    ], rows=1, savefig=f"{showpath}/{tag}_scan0.jpg", fontsize=60)
    visualize_attnmaps([
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao0", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao1", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao2", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao3", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs), ""),
    ], rows=2, savefig=f"{showpath}/{tag}_scan.jpg", fontsize=60)
    visualize_attnmaps([
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs)[0].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs)[int(H * W / 3)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs)[int(H * W / 3 * 2)].view(H, W),
         ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs)[-1].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs)[0].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs)[int(H * W / 3)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs)[int(H * W / 3 * 2)].view(H, W),
         ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs)[-1].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs)[0].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs)[int(H * W / 3)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs)[int(H * W / 3 * 2)].view(H, W),
         ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs)[-1].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs)[0].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs)[int(H * W / 3)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs)[int(H * W / 3 * 2)].view(H, W),
         ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs)[-1].view(H, W), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[0].view(H, W), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[int(H * W / 3)].view(H, W), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[int(H * W / 3 * 2)].view(H, W), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[-1].view(H, W), ""),
    ], rows=4, dpi=200, savefig=f"{showpath}/{tag}_scan_procedure.jpg", fontsize=100)


def visual_ws(As, Bs, Cs, Ds, us, dts, delta_bias, with_ws=False, with_dt=False, only_ws=False, ratio=1, tag="bcs",
                 H=56, W=56, front_point=(0.5, 0.5), front_back=(0.7, 0.8), showpath=os.path.join(this_path, "show")
                 , half=False):
    kwargs = dict(with_ws=with_ws, with_dt=with_dt, only_ws=only_ws, ratio=ratio)

    ave_ws_a0 = attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ave_ws_a0", half=half, **kwargs)
    ave_ws_a1 = attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ave_ws_a1", half=half, **kwargs)
    ave_ws_a2 = attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ave_ws_a2", half=half, **kwargs)
    ave_ws_a3 = attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ave_ws_a3", half=half, **kwargs)
    return ave_ws_a0.cpu(), ave_ws_a1.cpu(), ave_ws_a2.cpu(), ave_ws_a3.cpu()
def main():
    stages = [2,3]#[1, 2]
    example = 77165
    dataset_path = "/opt/dataset"
    model_name = 'vssm_nano'#, 'vssm_nano', 'vssm_ms_nano', 'vssm_ms'
    coco_backbone = True
    if model_name == 'vssm_ms':
        vssm: nn.Module = VSSM(
            depths=(1, 2, 9, 2),
            dims=96,
            out_indices=(0, 1, 2, 3),
            # pretrained="../exclude/ablation_logs/vssm_tiny_ms_e300/20240417035430/best_ckpt.pth",
            pretrained="../exclude/weights/vssm_tiny_ms_e300.pth",
            ssm_d_state=1,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=0.0,
            downsample_version="v1",
            patchembed_version="v1",
            sscore_type="multiscale_4scan_12",
            convFFN=True,
            add_se=True,
            ms_stage=[0, 1, 2, 3],
            ms_split=[1, 3],
            ffn_dropout=0.0,
        ).cuda().eval()
        dst_path = '/home/li/yh/mamba/vmamba/exclude/fig/attnmap'
        ckpt_path = "/home/li/yh/mamba/vmamba/exclude/weights/mask_rcnn_ms_vssm_fpn_coco_tiny.pth"
        if not coco_backbone:
            ckpt_path = "../exclude/weights/vssm_tiny_ms_e300.pth"
            dst_path = '/home/li/yh/mamba/vmamba/exclude/fig/attnmap_imagenet'
    elif model_name == 'vssm':
        vssm: nn.Module = VSSM(
            depths=(2, 2, 9, 2),
            dims=96,
            out_indices=(0, 1, 2, 3),
            ssm_d_state=16,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=0.0,
            downsample_version="v1",
            patchembed_version="v1",
        ).cuda().eval()
        ckpt_path = "/home/li/yh/mamba/vmamba/exclude/weights/tiny_maskrcnn_epoch_12.pth"
        dst_path = '/home/li/yh/mamba/vmamba/exclude/fig/attnmap_vmamba'
        if not coco_backbone:
            ckpt_path = "../exclude/weights/vssmtiny_dp01_ckpt_epoch_292.pth"
            dst_path = '/home/li/yh/mamba/vmamba/exclude/fig/attnmap_vmamba_imagenet'
    elif model_name == 'vssm_nano':
        vssm: nn.Module = VSSM(
            depths=(1, 2, 4, 2),
            dims=48,
            out_indices=(0, 1, 2, 3),
            ssm_d_state=8,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=0.0,
            downsample_version="v1",
            patchembed_version="v1",
        ).cuda().eval() #vssm nano version
        ckpt_path = "/home/li/yh/mamba/vmamba/exclude/weights/mask_rcnn_vssm_fpn_coco_0p9G.pth"
        dst_path = '/home/li/yh/mamba/vmamba/exclude/fig/attnmap_vmamba_nano'
        if not coco_backbone:
            ckpt_path = "/home/li/yh/mamba/vmamba/exclude/log/vssm_nano_224_0p9G/20240412175113/20240412175113/best_ckpt.pth"
            dst_path = '/home/li/yh/mamba/vmamba/exclude/fig/attnmap_vmamba_nano_imagenet'
    elif model_name == 'vssm_ms_nano':
        # ms vssm nano version -------------------------------------------------------
        vssm: nn.Module = VSSM(
            depths=(1, 2, 5, 2),
            dims=48,
            out_indices=(0, 1, 2, 3),
            # pretrained="../exclude/ablation_logs/vssm_tiny_ms_e300/20240417035430/best_ckpt.pth",
            pretrained="../exclude/weights/vssm_tiny_ms_e300.pth",
            ssm_d_state=1,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=0.0,
            downsample_version="v1",
            patchembed_version="v1",
            sscore_type="multiscale_4scan_12",
            convFFN=True,
            add_se=True,
            ms_stage=[0, 1, 2, 3],
            ms_split=[1, 3],
            ffn_dropout=0.0,
        ).cuda().eval()
        ckpt_path = "/home/li/yh/mamba/vmamba/exclude/weights/mask_rcnn_ms_vssm_fpn_coco_nano.pth"
        dst_path = '/home/li/yh/mamba/vmamba/exclude/fig/attnmap_nano'
        if not coco_backbone:
            ckpt_path = "../exclude/weights/vssm_nano_ms_e100.pth"
            dst_path = '/home/li/yh/mamba/vmamba/exclude/fig/attnmap_nano_imagenet'
    if coco_backbone:
        vssm.load_state_dict(convert_state_dict(torch.load(open(
            ckpt_path,
            "rb"), map_location="cpu")["state_dict"]), strict=False)
    else:
        vssm.load_state_dict(torch.load(open(
            ckpt_path,
            "rb"), map_location="cpu")["model"], strict=False)


    vssm, ss2ds = add_hook(vssm)

    data = get_dataloader(batch_size=32, root=dataset_path, sequential=True,
                          img_size=448)
    dataset = data.dataset
    #vis one example
    # img, label = dataset[example]
    # with torch.no_grad():
    #     out = vssm(img[None].cuda())
    # print(out.argmax().item(), label)
    # for stage in stages:
    #     dst_path = '/home/li/yh/mamba/vmamba/exclude/fig/attnmap_nano/example{}/stage{}'.format(example, stage)
    #     check_path(dst_path)
    #     denormalize(img).save(os.path.join(dst_path, "imori.jpg"))
    #     regs = getattr(ss2ds[stage][-1], "__data__")
    #     As, Bs, Cs, Ds = -torch.exp(regs["A_logs"].to(torch.float32)), regs["Bs"], regs["Cs"], regs["Ds"]
    #     us, dts, delta_bias = regs["us"], regs["dts"], regs["delta_bias"]
    #     ys, oy = regs["ys"], regs["y"]
    #     print(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)
    #     B, G, N, L = Bs.shape
    #     GD, N = As.shape
    #     D = GD // G
    #     H, W = int(math.sqrt(L)), int(math.sqrt(L))
    #     visual_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, with_ws=False, with_dt=False, only_ws=False, ratio=1, tag="bcs",
    #                  H=H, W=W, showpath=dst_path,half=True)

    #vis random selected examples
    #select 50 random examples for 0 to len(dataset[example])-1
    rand_examples = random.sample(range(len(dataset)), 50)
    if 'vmamba' in dst_path.split('/')[-1]:
        half = False
        title = 'SS2D'
    else:
        half = True
        title = 'MS2D'
    check_path(dst_path)
    ws_rec = {}
    for example in rand_examples:
        img, label = dataset[example]
        with torch.no_grad():
            out = vssm(img[None].cuda())
            for stage in stages:
                #check_path(dst_path)
                #denormalize(img).save(os.path.join(dst_path, "imori.jpg"))
                layers = len(ss2ds[stage])
                if 'stage{}'.format(stage) not in ws_rec:
                    ws_rec['stage{}'.format(stage)] = {}
                for layer in range(layers):
                    if 'layer{}'.format(layer) not in ws_rec['stage{}'.format(stage)]:
                        ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)] = {}
                    if 'ave_ws' not in ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]:
                        ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ave_ws'] = []
                    if 'ratio' not in ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]:
                        ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ratio'] = []
                    if 'ws_ah' not in ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]:
                        ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ws_ah'] = []
                    if 'ws_aw' not in ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]:
                        ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ws_aw'] = []

                    regs = getattr(ss2ds[stage][layer], "__data__")
                    As, Bs, Cs, Ds = -torch.exp(regs["A_logs"].to(torch.float32)), regs["Bs"], regs["Cs"], regs["Ds"]
                    us, dts, delta_bias = regs["us"], regs["dts"], regs["delta_bias"]
                    ys, oy = regs["ys"], regs["y"]
                    print(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)
                    B, G, N, L = Bs.shape
                    GD, N = As.shape
                    D = GD // G
                    H, W = int(math.sqrt(L)), int(math.sqrt(L))
                    ave_ws = visual_ws(As, Bs, Cs, Ds, us, dts, delta_bias, with_ws=False, with_dt=False, only_ws=False, ratio=1, tag="bcs",
                                 H=H, W=W, showpath=dst_path,half=half)

                    ave_ah, ave_aw = get_centra_attn_ratio(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, with_ws=True, with_dt=False, only_ws=True, ret="all",
                          ratio=1, verbose=False, half=half)
                    ave_ratio = torch.max(ave_ah.mean(0) / ave_aw.mean(0), ave_aw.mean(0) / ave_ah.mean(0)).cpu()
                    ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ave_ws'].append(ave_ws)
                    ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ratio'].append(ave_ratio)
                    ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ws_ah'].append(ave_ah.cpu())
                    ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ws_aw'].append(ave_aw.cpu())
                    del As, Bs, Cs, Ds, us, dts, delta_bias
                    del ave_ah, ave_aw
                    torch.cuda.empty_cache()


    # show decay along channel
    for stage in stages:
        layers = len(ss2ds[stage])
        for layer in range(layers):
            ws_ah = torch.stack(ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ws_ah']).mean(0).flatten(-2,-1)
            ws_aw = torch.stack(ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ws_aw']).mean(0).transpose(-1,-2).flatten(-2,-1)
            ws_ah_f,ws_ah_b = ws_ah[:,:L//2-W//2].mean(-1), ws_ah[:,L//2-W//2:].mean(-1)
            ws_aw_f,ws_aw_b = ws_aw[:,:L//2-H//2-1].mean(-1), ws_aw[:,L//2-H//2-1:].mean(-1)

            ah_f_sorted, ah_f_idx = torch.sort(ws_ah_f, descending=True)
            aw_f_sorted, aw_f_idx = torch.sort(ws_aw_f, descending=True)
            ah_b_sorted, ah_b_idx = torch.sort(ws_ah_b, descending=True)
            aw_b_sorted, aw_b_idx = torch.sort(ws_aw_b, descending=True)
            decay_dst_path = os.path.join(dst_path,'stage{}_layer{}'.format(stage, layer) + 'decay.jpg')
            vis_channel_decay(ah_f_sorted, aw_f_sorted, ah_b_sorted, aw_b_sorted, decay_dst_path)

            ave_ratio = torch.stack(ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ratio']).mean(0)
            binary_ave_ratio = torch.where(ave_ratio > 10, torch.ones_like(ave_ratio), torch.zeros_like(ave_ratio))
            visualize_attnmap(ave_ratio,dpi=100,savefig=f"{dst_path}/stage{stage}_layer{layer}_ave_ratio.jpg")
            visualize_attnmap(binary_ave_ratio, dpi=100, savefig=f"{dst_path}/stage{stage}_layer{layer}_binary_ave_ratio.jpg")
            visualize_attnmap(torch.stack(ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ws_ah']).mean(0).mean(0), dpi=100, savefig=f"{dst_path}/stage{stage}_layer{layer}_ave_ws_ah.jpg")
            visualize_attnmap(torch.stack(ws_rec['stage{}'.format(stage)]['layer{}'.format(layer)]['ws_aw']).mean(0).mean(0), dpi=100, savefig=f"{dst_path}/stage{stage}_layer{layer}_ave_ws_aw.jpg")

    # for stage in stages:
    #     ws_a0 = torch.stack([ws_rec['stage{}'.format(stage)][i][0] for i in range(len(rand_examples))]).mean(0)
    #     ws_a1 = torch.stack([ws_rec['stage{}'.format(stage)][i][1] for i in range(len(rand_examples))]).mean(0)
    #     ws_a2 = torch.stack([ws_rec['stage{}'.format(stage)][i][2] for i in range(len(rand_examples))]).mean(0)
    #     ws_a3 = torch.stack([ws_rec['stage{}'.format(stage)][i][3] for i in range(len(rand_examples))]).mean(0)
    #     if half:
    #         #interpolate a1, a2, a3 to a0 size
    #         h, w = ws_a0.shape
    #         ws_a1 = F.interpolate(ws_a1[None,None], size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    #         ws_a2 = F.interpolate(ws_a2[None,None], size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    #         ws_a3 = F.interpolate(ws_a3[None,None], size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    #
    #     visualize_attnmaps([
    #         (ws_a0, "Scan1"),
    #         (ws_a1, "Scan2"),
    #         (ws_a2, "Scan3"),
    #         (ws_a3, "Scan4"),
    #     ], rows=2, savefig=f"{dst_path}/stage{stage}_ws_scan.jpg", fontsize=60, title=title)




    # visual_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, with_ws=True, with_dt=False, only_ws=True, ratio=1, tag="ws", H=H,
    #              W=W, showpath=dst_path)


if __name__ == "__main__":
    main()
