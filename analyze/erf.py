import os
import sys
import time
from functools import partial
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class visualize:
    @staticmethod
    def get_colormap(name):
        import matplotlib as mpl
        """Handle changes to matplotlib colormap interface in 3.6."""
        try:
            return mpl.colormaps[name]
        except AttributeError:
            return mpl.cm.get_cmap(name)

    @staticmethod
    def visualize_attnmap(attnmap, savefig="", figsize=(18, 16), cmap=None, sticks=True, dpi=400, fontsize=35,
                          linewidth=2, **kwargs):
        import matplotlib.pyplot as plt
        if isinstance(attnmap, torch.Tensor):
            attnmap = attnmap.detach().cpu().numpy()
        plt.rcParams["font.size"] = fontsize
        plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        ax = plt.gca()
        im = ax.imshow(attnmap, cmap=cmap)
        # ax.set_title(title)
        if not sticks:
            ax.set_yticks([])
            ax.set_xticks([])
        cbar = ax.figure.colorbar(im, ax=ax)
        if savefig == "":
            plt.show(bbox_inches='tight')
        else:
            plt.savefig(savefig,bbox_inches='tight')
        plt.close()

    @staticmethod
    def visualize_attnmaps(attnmaps, savefig="", figsize=(18, 16), rows=1, cmap=None, dpi=400, fontsize=35, linewidth=2,
                           **kwargs):
        # attnmaps: [(map, title), (map, title),...]
        import math
        import matplotlib.pyplot as plt
        vmin = min([np.min((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        vmax = max([np.max((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        cols = math.ceil(len(attnmaps) / rows)
        plt.rcParams["font.size"] = fontsize
        figsize = (cols * figsize[0], rows * figsize[1])
        fig, axs = plt.subplots(rows, cols, squeeze=False, sharex="all", sharey="all", figsize=figsize, dpi=dpi)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(attnmaps):
                    image = np.zeros_like(image)
                    title = "pad"
                else:
                    image, title = attnmaps[idx]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                im = axs[i, j].imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
                axs[i, j].set_title(title)
                axs[i, j].set_yticks([])
                axs[i, j].set_xticks([])
                print(title, "max", np.max(image), "min", np.min(image), end=" | ")
            print("")
        plt.subplots_adjust(wspace=0.15, hspace=0.15)  # Adjust spacing
        axs[0, 0].figure.colorbar(im, ax=axs)
        fig_title = kwargs.get('title', "")

        #add title to the figure
        if fig_title != "":
            fig.suptitle(fig_title, fontsize=int(fontsize*1.5), fontweight='bold')
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig,bbox_inches='tight')
            print(f"save to {savefig}")
        plt.close()
        print("")

    @staticmethod
    def seanborn_heatmap(
            data, *,
            vmin=None, vmax=None, cmap=None, center=None, robust=False,
            annot=None, fmt=".2g", annot_kws=None,
            linewidths=0, linecolor="white",
            cbar=True, cbar_kws=None, cbar_ax=None,
            square=False, xticklabels="auto", yticklabels="auto",
            mask=None, ax=None,
            **kwargs
    ):
        from matplotlib import pyplot as plt
        from seaborn.matrix import _HeatMapper
        # Initialize the plotter object
        plotter = _HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
                              annot_kws, cbar, cbar_kws, xticklabels,
                              yticklabels, mask)

        # Add the pcolormesh kwargs here
        kwargs["linewidths"] = linewidths
        kwargs["edgecolor"] = linecolor

        # Draw the plot and return the Axes
        if ax is None:
            ax = plt.gca()
        if square:
            ax.set_aspect("equal")
        plotter.plot(ax, cbar_ax, kwargs)
        mesh = ax.pcolormesh(plotter.plot_data, cmap=plotter.cmap, **kwargs)
        return ax, mesh

    @classmethod
    def visualize_snsmap(cls, attnmap, savefig="", figsize=(18, 16), cmap=None, sticks=True, dpi=80, fontsize=35,
                         linewidth=2, **kwargs):
        import matplotlib.pyplot as plt
        if isinstance(attnmap, torch.Tensor):
            attnmap = attnmap.detach().cpu().numpy()
        plt.rcParams["font.size"] = fontsize
        plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        ax = plt.gca()
        _, mesh = cls.seanborn_heatmap(attnmap, xticklabels=sticks, yticklabels=sticks, cmap=cmap, linewidths=0,
                                       center=0, annot=False, ax=ax, cbar=False, annot_kws={"size": 24}, fmt='.2f')
        cb = ax.figure.colorbar(mesh, ax=ax)
        cb.outline.set_linewidth(0)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()

    @classmethod
    def visualize_snsmaps(cls, attnmaps, model_names, col_labels, savefig="", figsize=(18, 16), rows=1, cmap=None, sticks=True, dpi=80,
                          fontsize=35, linewidth=2, **kwargs):
        # attnmaps: [(map, title), (map, title),...]
        import math
        import matplotlib.pyplot as plt
        vmin = min([np.min((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        vmax = max([np.max((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        cols = math.ceil(len(attnmaps) / rows)
        plt.rcParams["font.size"] = fontsize
        figsize = (cols * figsize[0], rows * figsize[1])
        fig, axs = plt.subplots(rows, cols, squeeze=False, sharex="all", sharey="all", figsize=figsize, dpi=dpi)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(attnmaps):
                    image = np.zeros_like(image)
                    title = "pad"
                else:
                    image, title = attnmaps[idx]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                _, im = cls.seanborn_heatmap(image, xticklabels=sticks, yticklabels=sticks,
                                             vmin=vmin, vmax=vmax, cmap=cmap,
                                             center=0, annot=False, ax=axs[i, j],
                                             cbar=False, annot_kws={"size": 24}, fmt='.2f')
                axs[i, j].set_title(title)
                if i == rows - 1:  # Set x-labels on the last row
                    if j == cols - 1:
                        axs[i, j].set_xlabel(model_names[j % len(model_names)],fontsize=fontsize,fontweight='bold')
                    else:
                        axs[i, j].set_xlabel(model_names[j % len(model_names)], fontsize=fontsize)
                if j == 0:  # Set y-labels on the first column
                    axs[i, j].set_ylabel(col_labels[i],fontsize=fontsize)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing between subplots
        plt.tight_layout()  # Optimize layout to make it more compact
        cb = axs[0, 0].figure.colorbar(im, ax=axs, pad=0.02, shrink=1.0)
        cb.outline.set_linewidth(0)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig, bbox_inches='tight')  # Save with tight bounding box
        plt.close()


def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


def simpnorm(data):
    data = np.power(data, 0.25)
    data = data / np.max(data)
    return data


def get_rectangle(data, thresh):
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, h // 2):
        selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
        area_sum = np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
    return None, None


def get_input_grad(model, samples, square=True, bchw=True):
    outputs = model(samples)
    if not bchw: # bhwc
        outputs = outputs.permute(0, 3, 1, 2) # bchw
    out_size = outputs.size()
    if square:
        assert out_size[2] == out_size[3]
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map


def get_input_grad_avg(model: nn.Module, size=1024, data_path=".", num_images=50, norms=lambda x: x, bchw=True):
    import tqdm
    from torchvision import datasets, transforms
    from torch.utils.data import SequentialSampler, DataLoader, RandomSampler
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=transform)
    data_loader_val = DataLoader(dataset, sampler=RandomSampler(dataset), pin_memory=True)

    meter = AverageMeter()
    model.cuda().eval()
    for _, (samples, _) in tqdm.tqdm(enumerate(data_loader_val)):
        if meter.count == num_images:
            break
        samples = samples.cuda(non_blocking=True).requires_grad_()
        contribution_scores = get_input_grad(model, samples,bchw=bchw)
        if np.isnan(np.sum(contribution_scores)):
            print("got nan | ", end="")
            continue
        else:
            meter.update(contribution_scores)
    return norms(meter.avg)


def check_path(path: str):
    if not os.path.exists(path): os.makedirs(path)

def main0():
    dst_path = '/home/li/yh/mamba/vmamba/exclude/fig/erf'
    check_path(dst_path)
    vssm_size = 'tiny'
    num_imgs = 50
    showpath = os.path.join(dst_path, f"vssm_{vssm_size}.png")
    data_path = "/opt/dataset"
    results_before = []
    results_after = []

    # modes = ["resnet", "convnext", "intern", "swin", "hivit", "deit", "vssma6", "vssmaav1"]
    # modes = ["resnet", "convnext", "swin", "deit", "hivit", "vssmaav1"]
    modes = ["resnet", "convnext", "swin", "deit",'vssm', 'msvssm']
    sys.path.append('../classification/models')
    _build = import_abspy("models", f"{os.path.dirname(__file__)}/../classification")
    build_mmpretrain_models = _build.build_mmpretrain_models

    def vssm_backbone(model, permute=False):
        class Permute(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(x):
                return x.permute(0, 3, 1, 2)

        model.classifier = Permute() if permute else nn.Identity()
        return model

    def intern_backbone(model):
        def forward(self, x):
            x = self.patch_embed(x)
            x = self.pos_drop(x)

            for level in self.levels:
                x = level(x)
            return x.permute(0, 3, 1, 2)

        model.forward = partial(forward, model)
        return model

    def mmpretrain_backbone(model, with_norm=False):
        from mmpretrain.models import build_classifier, ImageClassifier, ConvNeXt, VisionTransformer, SwinTransformer
        if isinstance(model.backbone, ConvNeXt):
            model.backbone.gap_before_final_norm = False
        if isinstance(model.backbone, VisionTransformer):
            model.backbone.out_type = 'featmap'

        def forward_backbone(self: ImageClassifier, x):
            x = self.backbone(x)[-1]
            return x

        if not with_norm:
            setattr(model, f"norm{model.backbone.out_indices[-1]}", lambda x: x)
        model.forward = partial(forward_backbone, model)
        return model

    if "resnet" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        model_before = partial(build_mmpretrain_models, cfg="resnet50", ckpt=False, only_backbone=True,
                               with_norm=False)()
        model_after = partial(build_mmpretrain_models, cfg="resnet50", ckpt=True, only_backbone=True, with_norm=False)()
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm, num_images=num_imgs), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm, num_images=num_imgs), model_name)
        ])

    if "convnext" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        model_before = partial(build_mmpretrain_models, cfg="convnext_tiny", ckpt=False, only_backbone=True,
                               with_norm=False, )()
        model_after = partial(build_mmpretrain_models, cfg="convnext_tiny", ckpt=True, only_backbone=True,
                              with_norm=False, )()
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm, num_images=num_imgs), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm, num_images=num_imgs), model_name)
        ])

    if "intern" in modes:
        HOME = os.environ["HOME"].rstrip("/")
        model_name = ""
        print("intern ================================", flush=True)
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/InternImage/classification")
        sys.path.insert(0, specpath)
        import DCNv3
        _model = import_abspy("intern_image", f"{HOME}/OTHERS/InternImage/classification/models/")
        model = partial(_model.InternImage, core_op='DCNv3', channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32],
                        offset_scale=1.0, mlp_ratio=4., )
        model_before = intern_backbone(model())
        model_after = intern_backbone(model())
        ckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/others/internimage_t_1k_224.pth"
        model_after.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"],
                                    strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        sys.path = sys.path[1:]

    if "swin" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier, ConvNeXt, VisionTransformer, SwinTransformer
        model = dict(
            type='ImageClassifier',
            backbone=dict(
                type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.2),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=768,
                init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
                loss=dict(
                    type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                cal_acc=False),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ],
            train_cfg=dict(augments=[
                dict(type='Mixup', alpha=0.8),
                dict(type='CutMix', alpha=1.0)
            ]),
        )
        ckpt = "https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth"
        model["backbone"].update({"img_size": 1024})
        model_before = mmpretrain_backbone(build_classifier(model))
        model_after = mmpretrain_backbone(build_classifier(model))
        model_after.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm, num_images=num_imgs), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm, num_images=num_imgs), model_name)
        ])

    if "deit" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        model_before = partial(build_mmpretrain_models, cfg="deit_small", ckpt=False, only_backbone=True,
                               with_norm=False, )()
        model_after = partial(build_mmpretrain_models, cfg="deit_small", ckpt=True, only_backbone=True,
                              with_norm=False, )()
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm, num_images=num_imgs), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm, num_images=num_imgs), model_name)
        ])

    if "hivit" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        from mmpretrain.models.builder import MODELS
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier, HiViT, VisionTransformer, SwinTransformer
        from mmpretrain.models.backbones.vision_transformer import resize_pos_embed, to_2tuple, np

        @MODELS.register_module()
        class HiViTx(HiViT):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.num_extra_tokens = 0
                self.interpolate_mode = "bicubic"
                self.patch_embed.init_out_size = self.patch_embed.patches_resolution
                self._register_load_state_dict_pre_hook(self._prepare_abs_pos_embed)
                self._register_load_state_dict_pre_hook(
                    self._prepare_relative_position_bias_table)

            # copied from SwinTransformer, change absolute_pos_embed to pos_embed
            def _prepare_abs_pos_embed(self, state_dict, prefix, *args, **kwargs):
                name = prefix + 'pos_embed'
                if name not in state_dict.keys():
                    return

                ckpt_pos_embed_shape = state_dict[name].shape
                if self.pos_embed.shape != ckpt_pos_embed_shape:
                    from mmengine.logging import MMLogger
                    logger = MMLogger.get_current_instance()
                    logger.info(
                        'Resize the pos_embed shape from '
                        f'{ckpt_pos_embed_shape} to {self.pos_embed.shape}.')

                    ckpt_pos_embed_shape = to_2tuple(
                        int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
                    pos_embed_shape = self.patch_embed.init_out_size

                    state_dict[name] = resize_pos_embed(state_dict[name],
                                                        ckpt_pos_embed_shape,
                                                        pos_embed_shape,
                                                        self.interpolate_mode,
                                                        self.num_extra_tokens)

            def _prepare_relative_position_bias_table(self, state_dict, *args, **kwargs):
                del state_dict['backbone.relative_position_index']
                return SwinTransformer._prepare_relative_position_bias_table(self, state_dict, *args, **kwargs)

        model = dict(
            backbone=dict(
                ape=True,
                arch='tiny',
                drop_path_rate=0.05,
                img_size=224,
                rpe=True,
                type='HiViTx'),
            head=dict(
                cal_acc=False,
                in_channels=384,
                init_cfg=None,
                loss=dict(
                    label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
                num_classes=1000,
                type='LinearClsHead'),
            init_cfg=[
                dict(bias=0.0, layer='Linear', std=0.02, type='TruncNormal'),
                dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
            ],
            neck=dict(type='GlobalAveragePooling'),
            train_cfg=dict(augments=[
                dict(alpha=0.8, type='Mixup'),
                dict(alpha=1.0, type='CutMix'),
            ]),
            type='ImageClassifier')
        ckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/others/hivit-tiny-p16_8xb128_in1k/epoch_295.pth"
        model["backbone"].update({"img_size": 1024})
        model_before = mmpretrain_backbone(build_classifier(model))
        model_after = mmpretrain_backbone(build_classifier(model))
        model_after.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])

    if "vssma6" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
        ta6 = partial(_model.VSSM, dims=96, depths=[2, 2, 9, 2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0,
                      forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1",
                      norm_layer="ln2d")
        ckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmtiny/vssmtiny_dp01_ckpt_epoch_292.pth"
        model_before = vssm_backbone(ta6().cuda().eval())
        model_after = vssm_backbone(ta6().cuda().eval())
        model_after.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"],
                                    strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])

    if "vssmaav1" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
        taav1 = partial(_model.VSSM, dims=96, depths=[2, 2, 5, 2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0,
                        ssm_conv=3, ssm_conv_bias=False, forward_type="v05noz", mlp_ratio=4.0, downsample_version="v3",
                        patchembed_version="v2", norm_layer="ln2d")
        ckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230/vssm1_tiny_0230_ckpt_epoch_262.pth"
        model_before = vssm_backbone(taav1().cuda().eval())
        model_after = vssm_backbone(taav1().cuda().eval())
        model_after.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"],
                                    strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])

    if 'vssm' in modes:
        #showpath = os.path.join(dst_path, "vssm.png")

        model_name = ""
        print(f"{model_name} ================================", flush=True)
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")

        # size: nano
        if vssm_size == 'nano':
            taav1 = partial(_model.VSSM, dims=48, depths=[1, 2, 4, 2], ssm_d_state=8, ssm_dt_rank="auto", ssm_ratio=2.0,
                            mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1",)
            ckpt = "/home/li/yh/mamba/vmamba/exclude/log/vssm_nano_224_0p9G/20240412175113/20240412175113/best_ckpt.pth"
        elif vssm_size == 'tiny':
            taav1 = partial(_model.VSSM, dims=96, depths=[2, 2, 9, 2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0,
                            mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1",)
            ckpt = "/home/li/yh/mamba/vmamba/exclude/weights/vssmtiny_dp01_ckpt_epoch_292.pth"

        model_before = vssm_backbone(taav1().cuda().eval())
        model_after = vssm_backbone(taav1().cuda().eval())
        model_after.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"],
                                    strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm, bchw=False, num_images=num_imgs), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm, bchw=False, num_images=num_imgs), model_name)
        ])
        total_erf = results_after[-1][0].sum()
        center_hs, center_ws = [10, 25,50,100,150,200], [10,25,50,100,150,200]
        # print('-'*50)
        # for center_h, center_w in zip(center_hs, center_ws):
        #     center_erf_h = results_after[-1][0][:, 512-center_h:512+center_h].sum()
        #     center_erf_w = results_after[-1][0][512-center_w:512+center_w, :].sum()
        #     after_center_occ = center_erf_h + center_erf_w - results_after[-1][0][512-center_w:512+center_w, 512-center_h:512+center_h].sum()
        #     print('center h, center w:', center_h, center_w)
        #     print('center erf ratio:', after_center_occ / total_erf)
        # print('-'*50)
    if "msvssm" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
        if vssm_size == 'nano':
            taav1 = partial(_model.VSSM, dims=48, depths=[1, 2, 5, 2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0,
                            sscore_type="multiscale_4scan_12", mlp_ratio=0.0, downsample_version="v1",
                            patchembed_version="v1", convFFN=True, add_se=True, ms_stage=[0, 1, 2, 3], ms_split=[1, 3],
                            ffn_dropout = 0.0)
            ckpt = "/home/li/yh/mamba/vmamba/exclude/log/vssm_nano_224_ms_drop_n1_e100/20240414140611/20240414140611/best_ckpt.pth"
        elif vssm_size == 'tiny':
            taav1 = partial(_model.VSSM, dims=96, depths=[1, 2, 9, 2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0,
                            sscore_type="multiscale_4scan_12", mlp_ratio=0.0, downsample_version="v1",
                            patchembed_version="v1", convFFN=True, add_se=True, ms_stage=[0, 1, 2, 3], ms_split=[1, 3],
                            ffn_dropout = 0.0)
            ckpt = "/home/li/yh/mamba/vmamba/exclude/log/vssm_tiny_ms_e300/vssm_tiny_ms_e300/20240417035430/best_ckpt.pth"
        model_before = vssm_backbone(taav1().cuda().eval())
        model_after = vssm_backbone(taav1().cuda().eval())
        model_after.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"],
                                    strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm, bchw=False, num_images=num_imgs), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm,bchw=False, num_images=num_imgs), model_name)
        ])
        # print('-' * 50)
        # total_erf = results_after[-1][0].sum()
        # center_hs, center_ws = [10,25,50, 100, 150, 200], [10,25,50, 100, 150, 200]
        # for center_h, center_w in zip(center_hs, center_ws):
        #     center_erf_h = results_after[-1][0][:, 512 - center_h:512 + center_h].sum()
        #     center_erf_w = results_after[-1][0][512 - center_w:512 + center_w, :].sum()
        #     after_center_occ = center_erf_h + center_erf_w - results_after[-1][0][512 - center_w:512 + center_w,
        #                                                      512 - center_h:512 + center_h].sum()
        #     print('center h, center w:', center_h, center_w)
        #     print('center erf ratio:', after_center_occ / total_erf)
        # print('-' * 50)

    model_names = ["ResNet50", "ConvNeXt-T", "Swin-T", "DeiT-S", "VMamba-T", "MSVMamba-T"]
    col_labels = ["Before Training", "After Training"]
    visualize.visualize_snsmaps(
        results_before + results_after,model_names,col_labels, savefig=showpath, rows=2, sticks=False, figsize=(5, 5), cmap='RdYlGn',
        dpi=100, fontsize=30
    )


if __name__ == "__main__":
    main0()
