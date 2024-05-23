_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="../exclude/ablation_logs/vssm_micro_ms_e300/20240417131438/best_ckpt.pth",
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
        dims=64,
        depths=(1, 2, 5, 2),
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
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150),
    auxiliary_head=dict(in_channels=256, num_classes=150)
)
# train_dataloader = dict(batch_size=4) # as gpus=4

