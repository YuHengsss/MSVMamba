_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MM_VSSM',
        depths=(2, 2, 27, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        pretrained="../../ckpts/classification/outs/vssm/vssmsmall/vssmsmall_dp03_ckpt_epoch_238.pth",
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        # forward_type="v0", # if you want exactly the same
    ),
)

# train_dataloader = dict(batch_size=2) # as gpus=8

