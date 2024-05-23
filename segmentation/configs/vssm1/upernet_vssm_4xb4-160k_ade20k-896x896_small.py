_base_ = [
    '../vssm/upernet_swin_4xb4-160k_ade20k-896x896_small.py'
]
model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="/home/zjy/nodeHPC9/downstream/vssm1_small_0229/ckpt_epoch_222.pth",
        # copied from classification/configs/vssm/vssm_small_224.yaml
        dims=96,
        depths=(2, 2, 15, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v3noz",
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.3,
    ),)
train_dataloader = dict(batch_size=4) # as gpus=4

