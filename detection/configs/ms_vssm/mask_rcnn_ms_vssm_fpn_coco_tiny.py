_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MM_VSSM',
        depths=(1, 2, 9, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        #pretrained="../exclude/ablation_logs/vssm_tiny_ms_e300/20240417035430/best_ckpt.pth",
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
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
)

# train_dataloader = dict(batch_size=2) # as gpus=8

