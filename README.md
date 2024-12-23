
<div align="center">
<h1>MSVMamba </h1>
<h3>Multi-Scale VMamba: Hierarchy in Hierarchy Visual State Space Model</h3>

Paper: ([arXiv:2405.14174](https://arxiv.org/abs/2405.14174))
</div>



## Introduction
MSVMamba is a visual state space model that introduces a hierarchy in hierarchy design to the VMamba model. This repository contains the code for training and evaluating MSVMamba models on the ImageNet-1K dataset for image classification, COCO dataset for object detection, and ADE20K dataset for semantic segmentation.
For more information, please refer to our [paper](https://arxiv.org/abs/2405.14174).

<p align="center">
  <img src="./assets/ms2d.jpg" width="800" />
</p>

## Main Results

### **Classification on ImageNet-1K**

|       name       | pretrain | resolution | acc@1 | #params | FLOPs |                                               logs&ckpts                                                | 
|:----------------:| :---: | :---: |:-----:|:-------:|:-----:|:-------------------------------------------------------------------------------------------------------:|
| MSVMambav3-Tiny  | ImageNet-1K | 224x224 | 83.0  |   32M   | 5.0G  |                                              [log&ckpt]()                                              | 
| MSVMambav3-Small | ImageNet-1K | 224x224 | 84.1  |   50M   | 8.8G  |                                              [log&ckpt]()                                              | 
| MSVMambav3-Base  | ImageNet-1K | 224x224 | 84.4  |   91M   | 16.3G |                                              [log&ckpt]()                                              | 



## Getting Started
The steps to create env, train and evaluate MSVMamba models are followed by the same steps as VMamba.

### Installation

**Step 1: Clone the MSVMamba repository:**

```bash
git clone https://github.com/YuHengsss/MSVMamba.git
cd MSVMamba
```

**Step 2: Environment Setup:**

***Create and activate a new conda environment***

```bash
conda create -n msvmamba
conda activate msvmamba
```

***Install Dependencies***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```
<!-- cd kernels/cross_scan && pip install . -->


***Dependencies for `Detection` and `Segmentation` (optional)***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

<!-- conda create -n cu12 python=3.10 -y && conda activate cu12
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# install cuda121 for windows
# install https://visualstudio.microsoft.com/visual-cpp-build-tools/
pip install timm==0.4.12 fvcore packaging -->


### Quick Start

**Classification**

To train MSVMamba models for classification on ImageNet, use the following commands for different configurations:

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp
```

If you only want to test the performance (together with params and flops):

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp --resume </path/of/checkpoint> --eval
```

**Detection and Segmentation**

To evaluate with `mmdetection` or `mmsegmentation`:
```bash
bash ./tools/dist_test.sh </path/to/config> </path/to/checkpoint> 1
```
*use `--tta` to get the `mIoU(ms)` in segmentation*

To train with `mmdetection` or `mmsegmentation`:
```bash
bash ./tools/dist_train.sh </path/to/config> 8
```


## Citation
If MSVMamba is helpful for your research, please cite the following paper:
```
@article{shi2024multiscale,
      title={Multi-Scale VMamba: Hierarchy in Hierarchy Visual State Space Model}, 
      author={Yuheng Shi and Minjing Dong and Chang Xu},
      journal={arXiv preprint arXiv:2405.14174},
      year={2024}
}
```

## Acknowledgment

This project is based on VMamba([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)), Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Swin-Transformer ([paper](https://arxiv.org/pdf/2103.14030.pdf), [code](https://github.com/microsoft/Swin-Transformer)), ConvNeXt ([paper](https://arxiv.org/abs/2201.03545), [code](https://github.com/facebookresearch/ConvNeXt)), [OpenMMLab](https://github.com/open-mmlab),
 thanks for their excellent works.

