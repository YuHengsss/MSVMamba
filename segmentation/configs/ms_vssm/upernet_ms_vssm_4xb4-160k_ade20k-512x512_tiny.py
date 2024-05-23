_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="../exclude/ablation_logs/vssm_tiny_ms_e300/20240417035430/best_ckpt.pth",
        dims=96,
        depths=(1, 2, 9, 2),
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
        drop_path_rate=0.2,
    ),)
# train_dataloader = dict(batch_size=4) # as gpus=4

