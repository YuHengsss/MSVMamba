_base_ = [
    '../swin/swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="../../ckpts/classification/outs/vssm/vssmbasedp05/vssmbase_dp05_ckpt_epoch_260.pth",
        # copied from classification/configs/vssm/vssm_base_224.yaml
        dims=128,
        depths=(2, 2, 27, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        # forward_type="v0", # if you want exactly the same
    ),)
# train_dataloader = dict(batch_size=4) # as gpus=4

